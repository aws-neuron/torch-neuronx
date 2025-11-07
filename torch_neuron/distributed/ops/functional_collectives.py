import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group

from ..utils import get_reduce_type
from . import all_gather_op, all_reduce_op, reduce_scatter_op


def all_gather_into_tensor(input, group_size, group_name):
    assert input.size(0) != 0, "Cannot run all-gather on empty input tensor"

    group = _resolve_process_group(group_name)
    replica_groups = [dist.get_process_group_ranks(group)]
    assert group_size == len(replica_groups[0]), (
        f"group_size is not equal to the length of replica "
        f"groups, Expected {len(replica_groups[0])}, got {group_size}"
    )

    out_shape = list(input.shape)
    out_shape[0] = out_shape[0] * group_size
    # Calculate output stride to preserve input layout
    input_stride = input.stride()
    out_stride = list(input_stride)
    # First dimension stride stays the same (elements are concatenated)
    output_tensor = torch.empty_strided(out_shape, out_stride, device="neuron", dtype=input.dtype)

    all_gather_op(input, replica_groups, out=output_tensor)
    return output_tensor


def all_gather_into_tensor_out(input, group_size, group_name, output):
    assert input.size(0) != 0, "Cannot run all-gather on empty input tensor"

    group = _resolve_process_group(group_name)
    replica_groups = [dist.get_process_group_ranks(group)]
    assert group_size == len(replica_groups[0]), (
        f"group_size is not equal to the length of replica "
        f"groups, Expected {len(replica_groups[0])}, got {group_size}"
    )

    expected_shape = list(input.shape)
    expected_shape[0] = expected_shape[0] * group_size
    assert output.shape == tuple(expected_shape), (
        f"Output tensor shape mismatch. Expected {tuple(expected_shape)}, " f"got {output.shape}"
    )

    all_gather_op(input, replica_groups, out=output)
    return output


def all_gather_into_tensor_coalesced(inputs, group_size, group_name):
    assert len(inputs) > 0, "Cannot run all-gather on empty input list"
    assert all(inp.size(0) != 0 for inp in inputs), "Cannot run all-gather on empty input tensors"

    group = _resolve_process_group(group_name)
    replica_groups = [dist.get_process_group_ranks(group)]
    assert group_size == len(replica_groups[0]), (
        f"group_size is not equal to the length of replica "
        f"groups, Expected {len(replica_groups[0])}, got {group_size}"
    )

    output_tensors = []
    for inp in inputs:
        out_shape = list(inp.shape)
        out_shape[0] = out_shape[0] * group_size

        inp_stride = inp.stride()
        out_stride = list(inp_stride)

        output_tensors.append(
            torch.empty_strided(out_shape, out_stride, device="neuron", dtype=inp.dtype)
        )

    all_gather_op(inputs, replica_groups, out=output_tensors)

    return output_tensors


def all_reduce_(input, reduce_op, group_name):
    assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"
    assert input.numel() != 0, "Cannot run all-reduce_ on empty input tensor"
    group = _resolve_process_group(group_name)
    replica_groups = [dist.get_process_group_ranks(group)]

    reduce_type = reduce_op.upper() if isinstance(reduce_op, str) else get_reduce_type(reduce_op)

    all_reduce_op([input], replica_groups, reduce_type, out=(input,))
    return input


def all_reduce(input, reduce_op, group_name):
    assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"
    assert input.numel() != 0, "Cannot run all-reduce on empty input tensor"
    output = input.clone()
    return all_reduce_(output, reduce_op, group_name)


def reduce_scatter_tensor(input, reduce_op, group_size, group_name):
    assert input.size(0) != 0, "Cannot run reduce-scatter on empty input tensor"
    assert (
        input.size(0) % group_size == 0
    ), f"Input tensor size {input.size(0)} is not divisible by group size {group_size}"

    group = _resolve_process_group(group_name)
    replica_groups = [dist.get_process_group_ranks(group)]
    assert group_size == len(replica_groups[0]), (
        f"group_size is not equal to the length of replica "
        f"groups, Expected {len(replica_groups[0])}, got {group_size}"
    )

    # Output tensor size is input size divided by group size
    out_shape = list(input.shape)
    out_shape[0] = out_shape[0] // group_size
    input_stride = input.stride()
    out_stride = list(input_stride)
    # First dimension stride stays the same (elements are concatenated)
    output_tensor = torch.empty_strided(out_shape, out_stride, device="neuron", dtype=input.dtype)

    # Convert reduce_op string to the format expected by reduce_scatter_op

    reduce_type = reduce_op.upper() if isinstance(reduce_op, str) else get_reduce_type(reduce_op)

    reduce_scatter_op([input], replica_groups, reduce_type, out=output_tensor)
    return output_tensor


c10_lib = torch.library.Library("_c10d_functional", "IMPL")
c10_lib.impl("all_gather_into_tensor", all_gather_into_tensor, "PrivateUse1")
c10_lib.impl("all_gather_into_tensor_out", all_gather_into_tensor_out, "PrivateUse1")
c10_lib.impl("all_gather_into_tensor_coalesced", all_gather_into_tensor_coalesced, "PrivateUse1")
c10_lib.impl("reduce_scatter_tensor", reduce_scatter_tensor, "PrivateUse1")
c10_lib.impl("all_reduce", all_reduce, "PrivateUse1")
c10_lib.impl("all_reduce_", all_reduce_, "PrivateUse1")
