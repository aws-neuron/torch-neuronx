"""C++ NeuronBackend integration — Python HLO callback and PG creation."""

import os

import torch
from torch._C._distributed_c10d import OpType

from torch_neuronx._C import NeuronBackend
from torch_neuronx.python_ops.torch_mlir.ops.collectives import (
    get_all_gather_kernel,
    get_all_reduce_kernel,
    get_all_to_all_kernel,
    get_reduce_scatter_kernel,
)

_REDUCE_INT_TO_STR = {
    0: "SUM",
    1: "AVG",
    2: "PRODUCT",
    3: "MIN",
    4: "MAX",
    5: "BAND",
    6: "BOR",
    7: "BXOR",
    8: "PREMUL_SUM",
    9: "UNUSED",
}


def _format_replica_groups(replica_groups):
    """Encode replica_groups as a string TorchToNeuron's regex parses.

    Format: "[[r0,r1,...],[...],...]". TorchToNeuron::createReplicaGroups
    extracts the integer rows with `\\[([\\d,\\s]+)\\]`. Falls back to the
    WORLD_SIZE env var when the string contains no groups, but we always
    produce a valid spec so that path is never taken in practice.
    """
    return "[" + ",".join("[" + ",".join(str(r) for r in g) + "]" for g in replica_groups) + "]"


def _lower_all_reduce_torch_mlir(inputs, replica_groups, reduce_op, kwargs):
    """ALLREDUCE → torch-mlir trace of `_c10d_functional.all_reduce[_coalesced]`.

    Dynamic path: when the C++ NeuronBackend injects a runtime numel scalar, route to the
    dynamic all-reduce kernel lowering (threading the c10d reduce_op).
    """
    if os.environ.get("NEURON_DYN_NKI_COLLECTIVES") == "1" and is_dynamic_collective_inputs(inputs):
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives import (
            lower_dynamic_allreduce,
        )

        return lower_dynamic_allreduce(inputs, replica_groups, reduce_op)

    del kwargs
    reduce_op_str = _REDUCE_INT_TO_STR.get(reduce_op, "SUM").lower()

    group_name = _format_replica_groups(replica_groups)
    coalesced = len(inputs) > 1
    kernel = get_all_reduce_kernel(coalesced=coalesced)

    if coalesced:
        return kernel.lower(list(inputs), reduce_op=reduce_op_str, group_name=group_name)
    return kernel.lower(inputs[0], reduce_op=reduce_op_str, group_name=group_name)


def _lower_reduce_scatter_torch_mlir(inputs, replica_groups, reduce_op, kwargs):
    """REDUCE_SCATTER → torch-mlir trace of `_c10d_functional.reduce_scatter_tensor`."""
    reduce_op_str = _REDUCE_INT_TO_STR.get(reduce_op, "SUM").lower()

    out = kwargs["out"]
    out_tensor = out[0] if isinstance(out, list | tuple) else out

    group_name = _format_replica_groups(replica_groups)
    group_size = len(replica_groups[0])
    kernel = get_reduce_scatter_kernel()
    return kernel.lower(
        inputs[0],
        reduce_op=reduce_op_str,
        group_size=group_size,
        group_name=group_name,
        out_shape=tuple(out_tensor.shape),
    )


def _lower_all_to_all_torch_mlir(inputs, replica_groups, reduce_op, kwargs):
    """ALLTOALL → torch-mlir trace of `_c10d_functional.all_to_all_single`."""
    del reduce_op
    group_name = _format_replica_groups(replica_groups)
    group_size = len(replica_groups[0])
    kernel = get_all_to_all_kernel()
    return kernel.lower(list(inputs), group_size=group_size, group_name=group_name)


def is_dynamic_collective_inputs(inputs) -> bool:
    """Return whether inputs include an injected runtime element-count scalar.

    Args:
        inputs (Sequence[Any]): Candidate collective inputs.

    Returns:
        bool: ``True`` when there are exactly two inputs and the second is a
        one-element ``torch.int32`` tensor.

    Note:
        This identifies the runtime scalar injected by the C++ NeuronBackend
        for dynamic all-gather and all-reduce lowering.
    """
    return (
        len(inputs) == 2
        and isinstance(inputs[1], torch.Tensor)
        and inputs[1].dtype == torch.int32
        and inputs[1].numel() == 1
    )


def _lower_all_gather_torch_mlir(inputs, replica_groups, reduce_op, kwargs):
    """ALLGATHER → torch-mlir trace of `_c10d_functional.all_gather_into_tensor`.

    Dynamic path: when the C++ NeuronBackend injects a runtime numel scalar,
    route to the dynamic all-gather kernel lowering path.
    """
    del reduce_op

    # C++ only injects the numel scalar when the dynamic allgather is enabled
    if os.environ.get("NEURON_DYN_NKI_COLLECTIVES") == "1" and is_dynamic_collective_inputs(inputs):
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.dynamic_collectives import (
            lower_dynamic_allgather,
        )

        return lower_dynamic_allgather(inputs, replica_groups)

    slice_output = kwargs.get("slice_output", False)
    group_name = _format_replica_groups(replica_groups)
    group_size = len(replica_groups[0])
    coalesced = len(inputs) > 1
    kernel = get_all_gather_kernel(sliced=slice_output, coalesced=coalesced)
    if coalesced:
        return kernel.lower(list(inputs), group_size=group_size, group_name=group_name)
    return kernel.lower(inputs[0], group_size=group_size, group_name=group_name)


_OP_TYPE_TO_TORCH_MLIR = {
    OpType.ALLREDUCE: _lower_all_reduce_torch_mlir,
    OpType.REDUCE_SCATTER: _lower_reduce_scatter_torch_mlir,
    OpType.ALLTOALL: _lower_all_to_all_torch_mlir,
    OpType.ALLGATHER: _lower_all_gather_torch_mlir,
}


def lower_collective_hlo(op_type, inputs, replica_groups, reduce_op, kwargs):
    """Lower a collective with torch-MLIR after a C++ backend cache miss.

    Args:
        op_type (OpType): Collective operation to lower.
        inputs (Sequence[torch.Tensor]): Input tensors for the collective.
        replica_groups (list[list[int]]): Participating rank groups.
        reduce_op (int): ``c10d::ReduceOp`` enumeration value.
        kwargs (dict): Additional arguments supplied by C++.

    Returns:
        LoweringResult: Result containing ``stablehlo_bytes``, ``cache_key``,
        and ``has_collectives=True``.

    Raises:
        ValueError: If ``op_type`` has no registered lowering.

    Note:
        The C++ NeuronBackend invokes this callback with the GIL held.
    """
    lower_fn = _OP_TYPE_TO_TORCH_MLIR.get(op_type)
    if lower_fn is None:
        raise ValueError(f"Unsupported collective op_type: {op_type}")
    return lower_fn(inputs, replica_groups, reduce_op, kwargs)


def create_neuron_cpp_backend(store, options, rank, size, global_world_size):
    """Create a C++ NeuronBackend process group.

    Args:
        store (dist.Store): Process-group store, typically a prefix store,
            used to publish and resolve RDMA descriptors by global rank.
        options (Backend.Options): C++ backend options.
        rank (int): Rank within the process group.
        size (int): Number of ranks in the process group.
        global_world_size (int): Number of ranks across all process groups.

    Returns:
        NeuronBackend: Process group configured with the Python lowering callback.
    """
    return NeuronBackend(store, options, rank, size, global_world_size, lower_collective_hlo)
