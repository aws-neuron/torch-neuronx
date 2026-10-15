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
    """Gather rank tensors into a caller-provided output tensor.

    Args:
        input (torch.Tensor): Local tensor to gather from this rank.
        group_size (int): Collective rank count, accepted for schema compatibility.
        group_name (str): Name used to resolve the process group.
        out (torch.Tensor): Destination for rank-ordered gathered tensors.

    Returns:
        torch.Tensor: The supplied ``out`` tensor after launching the gather.

    Raises:
        RuntimeError: If native process-group device or shape validation fails.

    Note:
        Calls ``_allgather_base`` directly because ``NeuronPG`` lacks ``getBackend``,
        which the standard Torch implementation requires.
    """
    group = _resolve_process_group(group_name)
    opts = torch._C._distributed_c10d.AllgatherOptions()
    opts.asyncOp = False

    group._allgather_base(out, input, opts)
    return out


def all_gather_into_tensor(input, group_size, group_name):
    """Gather rank tensors into a new rank-ordered tensor.

    Args:
        input (torch.Tensor): Local tensor to gather from this rank.
        group_size (int): Number of ranks contributing tensors.
        group_name (str): Name used to resolve the process group.

    Returns:
        torch.Tensor: Gathered tensor with leading size ``input.shape[0] * group_size``.

    Raises:
        RuntimeError: If native process-group device or shape validation fails.

    Note:
        Uses ``asyncOp=False`` to avoid the CompositeExplicitAutograd fallback retaining
        tensor references in ``NeuronWork``'s future. Callers use the returned tensor and
        ``wait_tensor`` rather than ``future.value()``.
    """
    group = _resolve_process_group(group_name)
    output = input.new_empty(input.shape[0] * group_size, *input.shape[1:])

    opts = torch._C._distributed_c10d.AllgatherOptions()
    opts.asyncOp = False

    work = group._allgather_base(output, input, opts)
    torch._C._distributed_c10d._register_work(output, work)
    return output


def reduce_scatter_tensor(input, reduce_op, group_size, group_name):
    """Reduce rank tensors and scatter the leading dimension.

    Args:
        input (torch.Tensor): Tensor whose leading dimension contains all rank shards.
        reduce_op (str): Reduction key looked up in ``_REDUCE_OP_MAP``.
        group_size (int): Number of output shards along the leading dimension.
        group_name (str): Name used to resolve the process group.

    Returns:
        torch.Tensor: Reduced local shard with leading size ``input.shape[0] // group_size``.

    Raises:
        KeyError: If ``reduce_op`` is absent from ``_REDUCE_OP_MAP``.
        RuntimeError: If native process-group validation or submission fails.

    Note:
        Uses ``asyncOp=False`` to avoid retaining tensor references in ``NeuronWork``'s
        future; callers synchronize through the returned tensor and ``wait_tensor``.
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
    """Reduce and scatter multiple tensors in one collective.

    Args:
        inputs (list[torch.Tensor]): Tensors containing all rank shards on dimension zero.
        reduce_op (str): Reduction key looked up in ``_REDUCE_OP_MAP``.
        group_size (int): Number of output shards along dimension zero.
        group_name (str): Name used to resolve the process group.

    Returns:
        list[torch.Tensor]: Reduced local shards with leading dimensions divided by
        ``group_size``.

    Raises:
        KeyError: If ``reduce_op`` is absent from ``_REDUCE_OP_MAP``.
        RuntimeError: If native process-group validation or submission fails.

    Note:
        Runs with ``asyncOp=False`` and registers the collective work for every output.
    """
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
    """Reduce a cloned tensor across the resolved process group.

    Args:
        input (torch.Tensor): Tensor to clone and reduce without modification.
        reduce_op (str): Reduction key looked up in ``_REDUCE_OP_MAP``.
        group_name (str): Name used to resolve the process group.

    Returns:
        torch.Tensor: Contiguous reduced clone registered with the collective work.

    Raises:
        KeyError: If ``reduce_op`` is absent from ``_REDUCE_OP_MAP``.
        RuntimeError: If native process-group validation or submission fails.

    Note:
        Uses ``asyncOp=False`` and bypasses the CompositeExplicitAutograd fallback,
        whose Work/Future lifecycle can fail to synchronize with
        ``torch_neuronx.synchronize()`` and hang a rank.
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
    """Reduce cloned tensors together across the resolved process group.

    Args:
        inputs (list[torch.Tensor]): Tensors to clone and reduce without modification.
        reduce_op (str): Reduction key looked up in ``_REDUCE_OP_MAP``.
        group_name (str): Name used to resolve the process group.

    Returns:
        list[torch.Tensor]: Contiguous reduced clones of ``inputs``.

    Raises:
        KeyError: If ``reduce_op`` is absent from ``_REDUCE_OP_MAP``.
        RuntimeError: If native process-group validation or submission fails.

    Note:
        Uses ``asyncOp=False`` for the same Work/Future synchronization rationale as
        ``all_reduce`` and registers the returned work against each original input.
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
    """Register Neuron functional collective implementations.

    Note:
        Installs process-wide ``PrivateUse1`` implementations for gather,
        reduce-scatter, and all-reduce functional operators.
    """
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
