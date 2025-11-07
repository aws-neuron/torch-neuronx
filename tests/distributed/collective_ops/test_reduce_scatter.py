import os

import pytest
import torch
import torch.distributed as dist

from .base_collective_op import BaseCollectiveOpTest

WORLD_SIZE = 2


shape_test_variables = {
    "shape": [(10,), (10, 20), (10, 10, 10), (2, 4, 8, 10)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True, False],
    "async_op": [False],
    "group": [None],
}

scatter_dim_test_variables = {
    "shape": [(2, 4, 8, 10)],
    "scatter_dim": [0, 1, 2, 3],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [False],
    "group": [None],
}

op_test_variables = {
    "shape": [(2, 4, 8, 10)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM, dist.ReduceOp.AVG, dist.ReduceOp.MIN, dist.ReduceOp.MAX],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [False],
    "group": [None],
}

dtype_test_variables = {
    "shape": [(2, 4, 8, 10)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32, torch.float16, torch.int32, torch.bfloat16],
    "single_input": [True],
    "async_op": [False],
    "group": [None],
}

single_input_test_variables = {
    "shape": [(2, 4, 8, 10)],
    "scatter_dim": [2],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True, False],
    "async_op": [False],
    "group": [None],
}

async_op_test_variables = {
    "shape": [(2, 4, 8, 10)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [True, False],
    "group": [None],
}

group_test_variables = {
    "shape": [(2, 4, 8, 10)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [False],
    "group": [None, list(range(WORLD_SIZE)), [0]],
}

multi_group_test_variables = {
    "shape": [(4, 2)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [False],
    "group": [[0, 1, 2]],
}

boundary_condition_variables = {
    "shape": [(1,), (3,), (3, 4)],
    "scatter_dim": [0],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [False],
    "group": [None],
}

boundary_condition_variables_diff_scatter_dim = {
    "shape": [(1, 1), (3, 1), (3, 3), (2, 3)],
    "scatter_dim": [1],
    "op": [dist.ReduceOp.SUM],
    "dtype": [torch.float32],
    "single_input": [True],
    "async_op": [False],
    "group": [None],
}


def generate_test_params(variables):
    """Generate parametrize decorator dynamically.

    Args:
        variables (dict): Dictionary of parameter names and their possible values

    Returns:
        tuple: (param_names, param_values) for pytest.mark.parametrize
    """
    import itertools

    param_names = list(variables.keys())
    param_values = list(itertools.product(*variables.values()))
    return param_names, param_values


def run_min_reduce_scatter_test(rank, world_size, kwargs):
    """Test reduce_scatter with MIN operation."""
    # rank 0: [1, 2, 3, 4], rank 1: [3, 4, 5, 6]
    input_list = [
        torch.tensor(
            [rank * 2 + 1.0, rank * 2 + 2.0, rank * 2 + 3.0, rank * 2 + 4.0], dtype=torch.float32
        ).to("neuron")
    ]
    output = torch.zeros(2, dtype=torch.float32).to("neuron")

    # For MIN: rank 0 gets min([1,3], [2,4]) = [1,2], rank 1 gets min([3,5], [4,6]) = [3,4]
    expected = (
        torch.tensor([1.0, 2.0], dtype=torch.float32)
        if rank == 0
        else torch.tensor([3.0, 4.0], dtype=torch.float32)
    )

    dist.reduce_scatter(output=output, input_list=input_list, op=dist.ReduceOp.MIN)
    assert torch.allclose(output.cpu(), expected)


def run_max_reduce_scatter_test(rank, world_size, kwargs):
    """Test reduce_scatter with MAX operation."""
    # rank 0: [1, 2, 3, 4], rank 1: [3, 4, 5, 6]
    input_list = [
        torch.tensor(
            [rank * 2 + 1.0, rank * 2 + 2.0, rank * 2 + 3.0, rank * 2 + 4.0], dtype=torch.float32
        ).to("neuron")
    ]
    output = torch.zeros(2, dtype=torch.float32).to("neuron")

    # For MAX: rank 0 gets max([1,3], [2,4]) = [3,4], rank 1 gets max([3,5], [4,6]) = [5,6]
    expected = (
        torch.tensor([3.0, 4.0], dtype=torch.float32)
        if rank == 0
        else torch.tensor([5.0, 6.0], dtype=torch.float32)
    )

    dist.reduce_scatter(output=output, input_list=input_list, op=dist.ReduceOp.MAX)
    assert torch.allclose(output.cpu(), expected)


def run_reduce_scatter_avg_distributed_test(rank, world_size, kwargs):
    """Test reduce_scatter AVG distributed coordination."""
    input_list = [
        torch.tensor([rank + 1.0, rank + 2.0, rank + 3.0, rank + 4.0], dtype=torch.float32).to(
            "neuron"
        )
    ]
    output = torch.zeros(2, dtype=torch.float32).to("neuron")
    dist.reduce_scatter(output=output, input_list=input_list, op=dist.ReduceOp.AVG)

    # Calculate expected result for AVG operation
    # For rank 0: gets average of [1.0, 2.0] and [2.0, 3.0] = [1.5, 2.5]
    # For rank 1: gets average of [3.0, 4.0] and [4.0, 5.0] = [3.5, 4.5]
    if rank == 0:
        expected = torch.tensor([1.5, 2.5], dtype=torch.float32)
    else:  # rank == 1
        expected = torch.tensor([3.5, 4.5], dtype=torch.float32)

    assert output.device.type == "neuron"
    assert output.dtype == torch.float32
    assert output.shape == (2,)
    assert torch.allclose(
        output.cpu(), expected, rtol=1e-5
    ), f"Expected {expected}, got {output.cpu()}"


def unary_op_execute(tensor_a, tensor_b, op):
    """Execute unary operation on two tensors.

    Args:
        tensor_a (torch.Tensor): First tensor
        tensor_b (torch.Tensor): Second tensor
        op (str): Reduction operation

    Returns:
        torch.Tensor: Result of the operation
    """
    if op in {dist.ReduceOp.SUM, dist.ReduceOp.AVG}:
        return tensor_a + tensor_b  # Sum first, division happens later for AVG
    elif op == dist.ReduceOp.MIN:
        return torch.min(tensor_a, tensor_b)
    elif op == dist.ReduceOp.MAX:
        return torch.max(tensor_a, tensor_b)
    else:
        raise ValueError(f"Unsupported op: {op}")


def create_input_output_expected_for_single_tensor_list(rank, world_size, kwargs):
    """Create input, output, and expected tensors for single tensor test case.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters including shape, dtype, scatter_dim, etc.

    Returns:
        tuple: (input_list, output_tensor, expected_tensor)
    """
    input_tensor = torch.full(kwargs["shape"], rank + 1, dtype=kwargs["dtype"])
    input_list = [input_tensor.to("neuron")]
    output_tensor = torch.full(kwargs["shape"], rank + 1, dtype=kwargs["dtype"]).to("neuron")

    # Initialize expected tensor based on operation type
    if kwargs["op"] == dist.ReduceOp.MIN:
        expected = torch.full(kwargs["shape"], float("inf"), dtype=kwargs["dtype"])
    elif kwargs["op"] == dist.ReduceOp.MAX:
        expected = torch.full(kwargs["shape"], float("-inf"), dtype=kwargs["dtype"])
    else:
        expected = torch.full(kwargs["shape"], 0, dtype=kwargs["dtype"])

    ranks_to_include = kwargs["group"] if kwargs["group"] else list(range(world_size))
    world_size = len(ranks_to_include)

    for r in ranks_to_include:
        rank_tensor = torch.full(kwargs["shape"], r + 1, dtype=kwargs["dtype"])
        expected = unary_op_execute(expected, rank_tensor, kwargs["op"])

    # Apply division for AVG operation
    if kwargs["op"] == dist.ReduceOp.AVG:
        expected = expected / len(ranks_to_include)

    output_shape = list(kwargs["shape"])
    output_shape[kwargs["scatter_dim"]] //= world_size
    output_tensor = torch.full(output_shape, rank + 1, dtype=kwargs["dtype"]).to("neuron")

    slices = [slice(None)] * len(kwargs["shape"])
    slices[kwargs["scatter_dim"]] = slice(
        rank * (kwargs["shape"][kwargs["scatter_dim"]] // world_size),
        (rank + 1) * (kwargs["shape"][kwargs["scatter_dim"]] // world_size),
    )
    expected_tensor = input_tensor if rank not in ranks_to_include else expected[tuple(slices)]

    return input_list, output_tensor, expected_tensor


def create_input_output_expected_for_multiple_tensor_list(rank, world_size, kwargs):
    """Create input, output, and expected tensors for multiple tensor test case.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters including shape, dtype, op, etc.

    Returns:
        tuple: (input_list, output_tensor, expected_tensor)
    """
    input_tensor = torch.full(kwargs["shape"], rank + 1, dtype=kwargs["dtype"])
    input_list = [input_tensor.to("neuron")] * world_size
    output_tensor = torch.full(kwargs["shape"], rank + 1, dtype=kwargs["dtype"]).to("neuron")

    # Initialize expected tensor based on operation type
    if kwargs["op"] == dist.ReduceOp.MIN:
        expected_tensor = torch.full(kwargs["shape"], float("inf"), dtype=kwargs["dtype"])
    elif kwargs["op"] == dist.ReduceOp.MAX:
        expected_tensor = torch.full(kwargs["shape"], float("-inf"), dtype=kwargs["dtype"])
    else:
        expected_tensor = torch.full(kwargs["shape"], 0, dtype=kwargs["dtype"])

    ranks_to_include = kwargs["group"] if kwargs["group"] else list(range(world_size))
    world_size = len(ranks_to_include)

    for r in ranks_to_include:
        rank_tensor = torch.full(kwargs["shape"], r + 1, dtype=kwargs["dtype"])
        expected_tensor = unary_op_execute(expected_tensor, rank_tensor, kwargs["op"])

    # Apply division for AVG operation
    if kwargs["op"] == dist.ReduceOp.AVG:
        expected_tensor = expected_tensor / len(ranks_to_include)

    if rank not in ranks_to_include:
        expected_tensor = input_tensor

    return input_list, output_tensor, expected_tensor


def run_reduce_scatter(rank, world_size, kwargs):
    """Run reduce_scatter operation with given parameters.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters
    """
    if kwargs["single_input"]:
        input_list, output_tensor, expected_tensor = (
            create_input_output_expected_for_single_tensor_list(rank, world_size, kwargs)
        )
    else:
        input_list, output_tensor, expected_tensor = (
            create_input_output_expected_for_multiple_tensor_list(rank, world_size, kwargs)
        )

    group = dist.new_group(kwargs["group"]) if kwargs["group"] else None
    work = dist.reduce_scatter(
        output=output_tensor,
        input_list=input_list,
        op=kwargs["op"],
        async_op=kwargs["async_op"],
        group=group,
    )
    if kwargs["async_op"]:
        assert work is not None
        work.wait()
    assert torch.equal(output_tensor.cpu(), expected_tensor)


def run_large_tensor_test(rank, world_size, kwargs):
    """Test reduce_scatter with large tensors.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    large_size = 1000000
    input_list = [torch.ones(large_size).to("neuron")]  # 1M elements
    expected = torch.ones(int(large_size / world_size)) * world_size
    output_tensor = torch.ones(int(large_size / world_size)).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.to("cpu"), expected)


def run_invalid_op_test(rank, world_size, kwargs):
    """Test reduce_scatter with invalid operation. Expected to raise RuntimeError.

    Fails because: Passes string "invalid_op" instead of valid ReduceOp enum value.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    tensor_list = [torch.ones(4, 2).to("neuron")]
    output = torch.ones(4, 1).to("neuron")
    dist.reduce_scatter(output=output, input_list=tensor_list, op="invalid_op")


def run_with_inf_inputs_test(rank, world_size, kwargs):
    """Test reduce_scatter with infinity values.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    tensor_list = [torch.tensor([torch.inf]).to("neuron")] * world_size
    output = torch.tensor([torch.inf]).to("neuron")
    expected = torch.tensor([torch.inf])
    dist.reduce_scatter(output=output, input_list=tensor_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output.cpu(), expected)

    tensor_list = [torch.ones(4, 2).to("neuron") * torch.inf]
    output = torch.ones(4, 1).to("neuron")
    expected = torch.ones(4, 1) * torch.inf
    dist.reduce_scatter(output=output, input_list=tensor_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output.cpu(), expected)


def run_unsupported_reduce_op_test(rank, world_size, kwargs):
    """Test unsupported reduce operations like PROD, MIN, MAX.
    Expected to raise RuntimeError or NotImplementedError.

    Fails because: Uses ReduceOp.PROD which is not implemented in the neuron backend,
    only SUM operation is currently supported.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.PROD)


def run_mixed_dtype_test(rank, world_size, kwargs):
    """Test with mixed data types in input tensors. Expected to raise RuntimeError.

    Fails because: Input list contains tensors with different dtypes (float32 and int32),
    but all input tensors must have the same dtype for reduce operations.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [
        torch.ones(4, dtype=torch.float32).to("neuron"),
        torch.ones(4, dtype=torch.int32).to("neuron"),
    ]
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_shape_mismatch_test(rank, world_size, kwargs):
    """Test with incompatible input/output shapes.
    Expected to raise RuntimeError or ValueError.

    Fails because: Input tensor has size 8 but output tensor has size 3, which doesn't
    divide evenly by world_size (2), making scatter operation impossible.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(8).to("neuron")]
    output_tensor = torch.ones(3).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_concatenation_test(rank, world_size, kwargs):
    """Test multiple tensor concatenation along dimension 0.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(2, 4).to("neuron") * (i + 1) for i in range(world_size)]
    output_tensor = torch.ones(2, 4).to("neuron")
    expected = [input_list[0] * world_size, input_list[1] * world_size][rank]

    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected.cpu())


def run_replica_group_size_mismatch_test(rank, world_size, kwargs):
    """Test when scatter dimension doesn't match replica group size.
    Expected to raise RuntimeError.

    Fails because: Input size (6) divided by output size (2) gives factor 3, but the
    replica group size is 2 (world_size). The scatter factor must equal replica group size.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(6).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_non_neuron_device_test(rank, world_size, kwargs):
    """Test with tensors not on neuron device. Expected to raise RuntimeError.

    Fails because: All tensors are on CPU device, but neuron distributed operations
    require tensors to be on neuron device.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4)]
    output_tensor = torch.ones(2)
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_empty_tensor_list_test(rank, world_size, kwargs):
    """Test with empty tensor list. Expected to raise RuntimeError or IndexError.

    Fails because: Input list is empty, but reduce_scatter requires at least one
    input tensor to perform the operation.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = []
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_output_none_test(rank, world_size, kwargs):
    """Test when output parameter is None (should use input shape).
    Expected to raise RuntimeError.

    Fails because: Output parameter is None, but the torch implementation requires
    an explicit output tensor to be provided.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4).to("neuron")]
    result = dist.reduce_scatter(output=None, input_list=input_list, op=dist.ReduceOp.SUM)
    expected = torch.ones(2) * world_size
    assert torch.equal(result.cpu(), expected)


def run_identical_input_output_shapes_test(rank, world_size, kwargs):
    """Test when input and output have identical shapes (no scatter).
    Expected to raise RuntimeError or ValueError.

    Fails because: Input and output tensors have identical shapes (4,4), meaning no
    scattering dimension can be identified, which is invalid for reduce_scatter.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4, 4).to("neuron")]
    output_tensor = torch.ones(4, 4).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_high_dimensional_tensor_test(rank, world_size, kwargs):
    """Test with high-dimensional tensors.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(2, 3, 4, 5, 6).to("neuron")]
    output_tensor = torch.ones(1, 3, 4, 5, 6).to("neuron")
    expected_tensor = torch.ones(1, 3, 4, 5, 6) * world_size
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected_tensor)


def run_scatter_last_dimension_test(rank, world_size, kwargs):
    """Test scattering on the last dimension.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(3, 4, 6).to("neuron")]
    output_tensor = torch.ones(3, 4, 3).to("neuron")
    expected_tensor = torch.ones(3, 4, 3) * world_size
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected_tensor)


def run_dtype_consistency_test(rank, world_size, kwargs):
    """Test dtype consistency validation.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    tensor1 = torch.ones(4, dtype=torch.float32).to("neuron")
    tensor2 = torch.ones(4, dtype=torch.float16).to("neuron")
    input_list = [tensor1, tensor2]
    output_tensor = torch.ones(4).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_zero_dimension_test(rank, world_size, kwargs):
    """Test tensor with zero size in first dimension. Expected to raise RuntimeError.

    Fails because: One of the tensor dimensions has size 0, which creates invalid
    tensor shapes that cannot be processed by reduce_scatter operations.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters including zero_dim
    """
    x, y, z = 10, 10, 10
    outx, outy, outz = 10, 10, 10
    if kwargs["zero_dim"] == 0:
        x = 0
        outy = outy // world_size
    elif kwargs["zero_dim"] == 1:
        y = 0
        outx = outx // world_size
    elif kwargs["zero_dim"] == 2:
        z = 0
        outx = outx // world_size
    input_list = [torch.ones(x, y, z).to("neuron")]
    output_tensor = torch.ones(outx, outy, outz).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_missing_out_arg(rank, world_size, kwargs):
    """Test output shape inference when out is None. Expected to raise RuntimeError.

    Fails because: No output tensor is provided (output=None), but neuron backend
    requires explicit output tensor specification.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4, 6).to("neuron")]
    dist.reduce_scatter(input_list=input_list, op=dist.ReduceOp.SUM)


def run_out_shape_varied(rank, world_size, kwargs):
    """Test output shape inference with varied output shapes.
    Expected to raise RuntimeError.

    Fails because: Output tensor shape doesn't match expected scattered dimensions
    from input tensor, violating reduce_scatter shape requirements.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters including output_shape
    """
    input_list = [torch.ones(4, 6).to("neuron")]
    output_tensor = torch.ones(kwargs["output_shape"]).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_scatter_dimension_validation_test(rank, world_size, kwargs):
    """Test scatter dimension validation logic. Expected to raise RuntimeError.

    Fails because: Input size (7) cannot be evenly divided by output size (3),
    making it impossible to determine valid scatter dimensions.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(7).to("neuron")]
    output_tensor = torch.ones(3).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_replica_groups_test(rank, world_size, kwargs):
    """Test replica_groups access and calculation.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters including group
    """
    input_list, output_tensor, expected_tensor = (
        create_input_output_expected_for_single_tensor_list(rank, world_size, kwargs)
    )
    group = dist.new_group(kwargs["group"])
    dist.reduce_scatter(
        output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM, group=group
    )


def run_nan_values_test(rank, world_size, kwargs):
    """Test with NaN values."""
    input_list = [torch.tensor([float("nan"), 1.0, 2.0, 3.0]).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    expected = torch.full((4,), 0)

    for _r in range(world_size):
        rank_tensor = torch.tensor([float("nan"), 1.0, 2.0, 3.0])
        expected = unary_op_execute(expected, rank_tensor, dist.ReduceOp.SUM)
    expected = expected[: 4 // world_size] if rank == 0 else expected[4 // world_size :]
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    # compare only non nan parts, other parts are not comparable
    assert torch.equal(output_tensor.cpu()[1:], expected[1:])


def run_negative_values_test(rank, world_size, kwargs):
    """Test with negative values."""
    input_list = [torch.tensor([-1.0, -2.0, 3.0, 4.0]).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    expected = torch.full((4,), 0)

    for _r in range(world_size):
        rank_tensor = torch.tensor([-1.0, -2.0, 3.0, 4.0])
        expected = unary_op_execute(expected, rank_tensor, dist.ReduceOp.SUM)
    expected = expected[: 4 // world_size] if rank == 0 else expected[4 // world_size :]
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected)


def run_very_large_values_test(rank, world_size, kwargs):
    """Test with very large values."""
    large_val = 1e10
    input_list = [torch.full((4,), large_val).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    expected = torch.full((2,), large_val * world_size)
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected)


def run_very_small_values_test(rank, world_size, kwargs):
    """Test with very small values."""
    small_val = 1e-10
    input_list = [torch.full((4,), small_val).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    expected = torch.full((2,), small_val * world_size)
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected)


def run_mixed_positive_negative_test(rank, world_size, kwargs):
    """Test with mixed positive and negative values."""
    input_list = [torch.tensor([1.0, -1.0, 2.0, -2.0]).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    expected = torch.full((4,), 0)

    for _r in range(world_size):
        rank_tensor = torch.tensor([1.0, -1.0, 2.0, -2.0])
        expected = unary_op_execute(expected, rank_tensor, dist.ReduceOp.SUM)
    expected = expected[: 4 // world_size] if rank == 0 else expected[4 // world_size :]
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected)


def run_integer_overflow_test(rank, world_size, kwargs):
    """Test potential integer overflow scenarios.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    max_int = torch.iinfo(torch.int32).max
    input_list = [torch.full((4,), max_int, dtype=torch.int32).to("neuron")]
    output_tensor = torch.ones(2, dtype=torch.int32).to("neuron")
    expected = torch.full((4,), 0, dtype=torch.int32)

    for _r in range(world_size):
        rank_tensor = torch.full((4,), max_int, dtype=torch.int32)
        expected = unary_op_execute(expected, rank_tensor, dist.ReduceOp.SUM)
    expected = expected[: 4 // world_size] if rank == 0 else expected[4 // world_size :]
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_precision_test(rank, world_size, kwargs):
    """Test floating point precision scenarios."""
    # Test with values that might lose precision
    input_list = [torch.tensor([1.0000001, 1.0000002, 1.0000003, 1.0000004]).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    expected = torch.full((4,), 0)

    for _r in range(world_size):
        rank_tensor = torch.tensor([1.0000001, 1.0000002, 1.0000003, 1.0000004])
        expected = unary_op_execute(expected, rank_tensor, dist.ReduceOp.SUM)
    expected = expected[: 4 // world_size] if rank == 0 else expected[4 // world_size :]
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected)


def run_memory_layout_test(rank, world_size, kwargs):
    """Test different memory layouts.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_tensor = torch.ones(4, 8).to("neuron")
    non_contiguous = input_tensor.t()
    input_list = [non_contiguous]
    output_tensor = torch.ones(8, 2).to("neuron")
    expected = torch.full((8, 4), 0)

    for _r in range(world_size):
        rank_tensor = torch.ones(8, 4)
        expected = unary_op_execute(expected, rank_tensor, dist.ReduceOp.SUM)
    expected = expected[:, : 4 // world_size] if rank == 0 else expected[:, 4 // world_size :]
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)
    assert torch.equal(output_tensor.cpu(), expected)


def run_multiple_scatter_dim_test(rank, world_size, kwargs):
    """Test when multiple dimensions differ between input and output.
    Expected to raise RuntimeError or ValueError.

    Fails because: Multiple dimensions differ between input (4,6) and output (2,3),
    but reduce_scatter can only scatter along a single dimension.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4, 6).to("neuron")]
    output_tensor = torch.ones(2, 3).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_list_of_tensors_more_than_rg_loop(rank, world_size, kwargs):
    """Test enumerate tensors loop in concatenate_tensors.
    Expected to raise RuntimeError.

    Fails because: Input list contains 3 tensors but world_size is 2, creating a
    mismatch between number of input tensors and expected replica group size.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [
        torch.ones(2, 3).to("neuron"),
        torch.ones(2, 3).to("neuron"),
        torch.ones(2, 3).to("neuron"),
    ]
    output_tensor = torch.ones(2, 3).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_all_tensor_device_check_test(rank, world_size, kwargs):
    """Test all(tensor.device.type == 'neuron') check. Expected to raise RuntimeError.

    Fails because: Input list contains tensors on different devices (neuron and CPU),
    but all tensors must be on the same neuron device."""
    input_list = [torch.ones(4).to("neuron"), torch.ones(4)]
    output_tensor = torch.ones(4).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_all_tensor_size_check_test(rank, world_size, kwargs):
    """Test all(tensor.size(0) != 0) check. Expected to raise RuntimeError.

    Fails because: One input tensor has size 0 in the first dimension, which
    violates the requirement that all tensors must have non-zero first dimension."""
    input_list = [torch.ones(4).to("neuron"), torch.ones(0, 4).to("neuron")]
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_all_tensor_dtype_check_test(rank, world_size, kwargs):
    """Test all(tensor.dtype == first_dtype) check. Expected to raise RuntimeError.

    Fails because: Input tensors have different dtypes (float32 and int32), but
    all input tensors must have the same dtype for reduce operations."""
    input_list = [
        torch.ones(4, dtype=torch.float32).to("neuron"),
        torch.ones(4, dtype=torch.int32).to("neuron"),
    ]
    output_tensor = torch.ones(4).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_nested_list_validation_test(rank, world_size, kwargs):
    """Test that nested lists of tensors are rejected. Expected to raise RuntimeError.

    Fails because: Input is a nested list [[tensor, tensor]] instead of flat list,
    but reduce_scatter expects a flat list of tensors."""
    nested_input = [[torch.ones(4).to("neuron"), torch.ones(4).to("neuron")]]
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=nested_input, op=dist.ReduceOp.SUM)


def run_mixed_tensor_integer_validation_test(rank, world_size, kwargs):
    """Test that mixed tensor and integer inputs are rejected.
    Expected to raise RuntimeError.

    Fails because: Input list contains both tensor and integer (42), but all
    elements must be tensors for reduce_scatter operation."""
    mixed_input = [torch.ones(4).to("neuron"), 42]
    output_tensor = torch.ones(2).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=mixed_input, op=dist.ReduceOp.SUM)


def run_tensor_shape_list_test(rank, world_size, kwargs):
    """Test tensor shape list conversion in concatenate_tensors.
    Expected to raise RuntimeError.

    Fails because: Input tensors have different shapes (2,3,4) and (2,3,6), but
    all input tensors must have identical shapes for concatenation.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(2, 3, 4).to("neuron"), torch.ones(2, 3, 6).to("neuron")]
    output_tensor = torch.ones(2, 3, 4).to("neuron")
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


def run_input_output_dtype_mismatch_test(rank, world_size, kwargs):
    """Test with different dtypes between input and output tensors.
    Expected to raise RuntimeError.

    Fails because: Input tensor is float32 but output tensor is float16, and
    input/output tensors must have matching dtypes for reduce_scatter.

    Args:
        rank (int): Current process rank
        world_size (int): Total number of processes
        kwargs (dict): Test parameters (unused)
    """
    input_list = [torch.ones(4, dtype=torch.float32).to("neuron")]
    output_tensor = torch.ones(2, dtype=torch.float16).to("neuron")  # Different dtype
    dist.reduce_scatter(output=output_tensor, input_list=input_list, op=dist.ReduceOp.SUM)


class TestReduceScatter(BaseCollectiveOpTest):
    """Test cases for torch.distributed.reduce_scatter."""

    def _run_test_with_params(self, **kwargs):
        """Common test logic for all parameter tests."""
        print(f"Running test with {kwargs}")
        try:
            if kwargs["scatter_dim"] < len(kwargs["shape"]):
                self.distributed_tester.run_test(run_reduce_scatter, **kwargs)
                print(f"✓ PASSED: {kwargs}")
            else:
                print(f"✓ IGNORED: scatter_dim >= len(shape) for {kwargs}")
        except Exception as e:
            print(f"✗ Possible Failure (Negative tests hit this also): {kwargs} - {e}")
            raise

    # ========================================================================
    # PARAMETER VALIDATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        ",".join(generate_test_params(shape_test_variables)[0]),
        generate_test_params(shape_test_variables)[1],
    )
    def test_shape(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with different shapes."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    @pytest.mark.parametrize(
        ",".join(generate_test_params(scatter_dim_test_variables)[0]),
        generate_test_params(scatter_dim_test_variables)[1],
    )
    def test_scatter_dim(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with different scatter dimensions."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    @pytest.mark.parametrize(
        ",".join(generate_test_params(op_test_variables)[0]),
        generate_test_params(op_test_variables)[1],
    )
    def test_op(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with different operations."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    def test_min_reduce_scatter(self):
        """Test reduce_scatter with MIN operation."""
        self.distributed_tester.run_test(run_min_reduce_scatter_test)

    def test_max_reduce_scatter(self):
        """Test reduce_scatter with MAX operation."""
        self.distributed_tester.run_test(run_max_reduce_scatter_test)

    def test_reduce_scatter_avg_distributed(self):
        """Test reduce_scatter AVG distributed coordination."""
        self.distributed_tester.run_test(run_reduce_scatter_avg_distributed_test)

    @pytest.mark.parametrize(
        ",".join(generate_test_params(dtype_test_variables)[0]),
        generate_test_params(dtype_test_variables)[1],
    )
    def test_dtype(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with different data types."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    @pytest.mark.parametrize(
        ",".join(generate_test_params(single_input_test_variables)[0]),
        generate_test_params(single_input_test_variables)[1],
    )
    def test_single_input(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with single vs multiple input tensors."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    @pytest.mark.parametrize(
        ",".join(generate_test_params(async_op_test_variables)[0]),
        generate_test_params(async_op_test_variables)[1],
    )
    def test_async_op(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with async operations."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    @pytest.mark.parametrize(
        ",".join(generate_test_params(group_test_variables)[0]),
        generate_test_params(group_test_variables)[1],
    )
    def test_group(self, shape, scatter_dim, op, dtype, single_input, async_op, group):
        """Test with different process groups."""
        self._run_test_with_params(
            shape=shape,
            scatter_dim=scatter_dim,
            op=op,
            dtype=dtype,
            single_input=single_input,
            async_op=async_op,
            group=group,
        )

    # ========================================================================
    # TENSOR DIMENSION & SHAPE TESTS
    # ========================================================================

    def test_high_dimensional_tensors(self):
        """Test with high-dimensional tensors."""
        self.distributed_tester.run_test(run_high_dimensional_tensor_test)

    def test_scatter_last_dimension(self):
        """Test scattering on the last dimension."""
        self.distributed_tester.run_test(run_scatter_last_dimension_test)

    @pytest.mark.parametrize(
        ",".join(generate_test_params(boundary_condition_variables)[0]),
        generate_test_params(boundary_condition_variables)[1],
    )
    def test_boundary_conditions(
        self, shape, scatter_dim, op, dtype, single_input, async_op, group
    ):
        """Test boundary conditions with world_size related dimensions."""
        with pytest.raises(RuntimeError):
            self._run_test_with_params(
                shape=shape,
                scatter_dim=scatter_dim,
                op=op,
                dtype=dtype,
                single_input=single_input,
                async_op=async_op,
                group=group,
            )

    @pytest.mark.parametrize(
        ",".join(generate_test_params(boundary_condition_variables_diff_scatter_dim)[0]),
        generate_test_params(boundary_condition_variables_diff_scatter_dim)[1],
    )
    def test_boundary_conditions_diff_scatter_dim(
        self, shape, scatter_dim, op, dtype, single_input, async_op, group
    ):
        """Test boundary conditions with world_size related dimensions."""
        with pytest.raises(RuntimeError):
            self._run_test_with_params(
                shape=shape,
                scatter_dim=scatter_dim,
                op=op,
                dtype=dtype,
                single_input=single_input,
                async_op=async_op,
                group=group,
            )

    @pytest.mark.parametrize("zero_dim", [0, 1, 2])
    def test_zero_dimensions(self, zero_dim):
        """Test tensor with zero size in first dimension."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_zero_dimension_test, zero_dim=zero_dim)

    def test_identical_shapes(self):
        """Test when input and output have identical shapes."""
        with pytest.raises((RuntimeError, ValueError)):
            self.distributed_tester.run_test(run_identical_input_output_shapes_test)

    def test_shape_mismatch(self):
        """Test with incompatible input/output shapes."""
        with pytest.raises((RuntimeError, ValueError)):
            self.distributed_tester.run_test(run_shape_mismatch_test)

    # ========================================================================
    # DATA TYPE & VALUE TESTS
    # ========================================================================

    def test_nan_values(self):
        """Test with NaN values."""
        self.distributed_tester.run_test(run_nan_values_test)

    def test_negative_values(self):
        """Test with negative values."""
        self.distributed_tester.run_test(run_negative_values_test)

    def test_very_large_values(self):
        """Test with very large values."""
        self.distributed_tester.run_test(run_very_large_values_test)

    def test_very_small_values(self):
        """Test with very small values."""
        self.distributed_tester.run_test(run_very_small_values_test)

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values."""
        self.distributed_tester.run_test(run_mixed_positive_negative_test)

    def test_floating_point_precision(self):
        """Test floating point precision scenarios."""
        self.distributed_tester.run_test(run_precision_test)

    @pytest.mark.xfail(
        reason="Integer overflow is not supported, currently scatters the input without "
        "reduce op, manual addition causes overflow, thus removing the assert"
    )
    def test_integer_overflow_scenarios(self):
        """Test potential integer overflow scenarios."""
        self.distributed_tester.run_test(run_integer_overflow_test)

    def test_with_inf_inputs(self):
        """Test reduce_scatter with infinity values."""
        self.distributed_tester.run_test(run_with_inf_inputs_test)

    def test_mixed_dtype_tensors(self):
        """Test with mixed data types in input tensors."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_mixed_dtype_test)

    def test_dtype_consistency_validation(self):
        """Test dtype consistency validation."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_dtype_consistency_test)

    def test_input_output_dtype_mismatch(self):
        """Test with different dtypes between input and output tensors."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_input_output_dtype_mismatch_test)

    # ========================================================================
    # DEVICE & MEMORY TESTS
    # ========================================================================

    def test_non_neuron_device(self):
        """Test with tensors not on neuron device."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_non_neuron_device_test)

    def test_memory_layout_variations(self):
        """Test different memory layouts."""
        self.distributed_tester.run_test(run_memory_layout_test)

    def test_large_tensor(self):
        """Test reduce_scatter with a large tensor."""
        self.distributed_tester.run_test(run_large_tensor_test)

    # ========================================================================
    # INPUT VALIDATION TESTS
    # ========================================================================

    def test_empty_tensor_list(self):
        """Test with empty tensor list."""
        with pytest.raises((RuntimeError, IndexError)):
            self.distributed_tester.run_test(run_empty_tensor_list_test)

    def test_output_none(self):
        """Test when output parameter is None."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_output_none_test)

    def test_out_missing(self):
        """Test output shape inference logic."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_missing_out_arg)

    def test_all_tensor_device_validation(self):
        """Test all tensor device validation."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_all_tensor_device_check_test)

    def test_all_tensor_size_validation(self):
        """Test all tensor size validation."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_all_tensor_size_check_test)

    def test_all_tensor_dtype_validation(self):
        """Test all tensor dtype validation."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_all_tensor_dtype_check_test)

    def test_nested_list_validation(self):
        """Test that nested lists of tensors are rejected."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_nested_list_validation_test)

    def test_mixed_tensor_integer_validation(self):
        """Test that mixed tensor and integer inputs are rejected."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_mixed_tensor_integer_validation_test)

    # ========================================================================
    # OPERATION & ALGORITHM TESTS
    # ========================================================================

    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_invalid_op_test)

    def test_unsupported_reduce_ops(self):
        """Test unsupported reduce operations."""
        with pytest.raises((RuntimeError, NotImplementedError)):
            self.distributed_tester.run_test(run_unsupported_reduce_op_test)

    def test_concatenation_logic(self):
        """Test multiple tensor concatenation logic."""
        self.distributed_tester.run_test(run_concatenation_test)

    def test_list_of_tensors_more_than_rg_loop(self):
        """Test enumerate tensors loop."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_list_of_tensors_more_than_rg_loop)

    def test_replica_group_size_mismatch(self):
        """Test scatter dimension vs replica group size mismatch."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_replica_group_size_mismatch_test)

    @pytest.mark.parametrize(
        ", ".join(generate_test_params(multi_group_test_variables)[0]),
        generate_test_params(multi_group_test_variables)[1],
    )
    def test_multiple_replica_groups(
        self, shape, scatter_dim, op, dtype, single_input, async_op, group
    ):
        """Test multiple replica groups."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(
                run_replica_groups_test,
                shape=shape,
                scatter_dim=scatter_dim,
                op=op,
                dtype=dtype,
                single_input=single_input,
                async_op=async_op,
                group=group,
            )

    def test_scatter_dimension_validation(self):
        """Test scatter dimension validation logic."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_scatter_dimension_validation_test)

    # ========================================================================
    # EDGE CASES & CORNER CASES
    # ========================================================================

    def test_multiple_scatter_dim(self):
        """Test multiple diff_indices case."""
        with pytest.raises((RuntimeError, ValueError)):
            self.distributed_tester.run_test(run_multiple_scatter_dim_test)

    # ========================================================================
    # EXPERIMENTAL/UNSUPPORTED FEATURES
    # ========================================================================

    @pytest.mark.xfail(
        reason="Less common scenario - need to think how to support as XLA RS "
        "scatter_dim needs special handling"
    )
    @pytest.mark.parametrize("output_shape", [(2, 2, 4, 6), (2, 2), (1, 3, 7, 9)])
    def test_out_shape_different_than_input(self, output_shape):
        """Test output shape different than input logic.
        This is supported by GPUs, GPUs reduce elementwise and
        then try to populate the overall reduced tensor starting from rank 0 to all other ranks,
        rows first and then once empty and nothing to send out, everything else is filled with
        zeros, thus no matter the output shape GPUs never fail.
        """
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_out_shape_varied, output_shape=output_shape)

    @pytest.mark.xfail(reason="Not supported yet, need to check on CUDA also")
    def test_tensor_diff_shapes(self):
        """Test tensor shape list conversion in concatenate_tensors."""
        with pytest.raises(RuntimeError):
            self.distributed_tester.run_test(run_tensor_shape_list_test)
