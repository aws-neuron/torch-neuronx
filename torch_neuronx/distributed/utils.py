import os
import socket

import torch
import torch.distributed as dist

from torch_neuronx.python_ops.base import _create_and_raise_detailed_error
from torch_neuronx.utils import flatten_tensors as _flatten_tensors

# Bucket size limit for collective operations (in MB)
_COLLECTIVE_BUCKETSIZE_BYTES: int = (
    int(os.environ.get("COLLECTIVE_BUCKETSIZE_IN_MB", "512")) * 1024 * 1024
)


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


def get_reduce_type(reduce_op):
    if reduce_op == dist.ReduceOp.SUM:
        return "SUM"
    elif reduce_op == dist.ReduceOp.PRODUCT:
        return "PRODUCT"
    elif reduce_op == dist.ReduceOp.MIN:
        return "MIN"
    elif reduce_op == dist.ReduceOp.MAX:
        return "MAX"
    elif reduce_op == dist.ReduceOp.BAND:
        return "BAND"
    elif reduce_op == dist.ReduceOp.BOR:
        return "BOR"
    elif reduce_op == dist.ReduceOp.BXOR:
        return "BXOR"
    elif reduce_op == dist.ReduceOp.AVG:
        return "AVG"
    elif reduce_op == dist.ReduceOp.PREMUL_SUM:
        return "PREMUL_SUM"
    else:
        raise ValueError(f"Unsupported reduce operation: {reduce_op}")


def create_global_replica_group(world_size, current_group_ranks):
    """
    This helper function will be deprecated once compiler and collectives
    team fully support partial groups for all to all.
    """
    group_size = len(current_group_ranks)
    unused_ranks = [i for i in range(world_size) if i not in current_group_ranks]
    global_replica_groups = [current_group_ranks] + [
        unused_ranks[j : j + group_size] for j in range(0, world_size - group_size, group_size)
    ]
    return global_replica_groups


def parse_and_update_inputs(inputs, options):
    """
    Preprocess inputs for PREMUL_SUM reduction operations by applying the
    premultiplication factor and converting to a regular SUM operation.

    For non-PREMUL_SUM operations, inputs are returned unchanged with their
    corresponding operation type.

    Args:
            inputs: Input tensors to process
            options: Options object with reduceOp attribute
    """
    reduce_type = get_reduce_type(options.reduceOp)
    if reduce_type == "PREMUL_SUM":
        _, premul_factor = options.reduceOp.__getstate__()
        if isinstance(inputs, list):
            if isinstance(inputs[0], list):
                scaled_inputs = [
                    [tensor * premul_factor for tensor in inner_list] for inner_list in inputs
                ]
            else:
                scaled_inputs = [tensor * premul_factor for tensor in inputs]
        else:
            scaled_inputs = inputs * premul_factor
        options.reduceOp = dist.ReduceOp.SUM
        return scaled_inputs, "SUM"

    return inputs, reduce_type


def _calculate_allgather_splits(tensor, world_size, max_output_bytes=512 * 1024 * 1024):
    """Split flattened input tensor to ensure allgather output doesn't exceed max size.

    Args:
        tensor: Tensor to flatten and split
        world_size: Number of ranks in the process group
        max_output_bytes: Maximum output size in bytes (default: 512MB)

    Returns:
        Tuple of flattened tensor splits
    """
    import math

    # Flatten tensor first to handle any shape (including batch size 1)
    flattened = tensor.flatten()

    total_output_bytes = flattened.numel() * tensor.element_size() * world_size
    num_splits = math.ceil(total_output_bytes / max_output_bytes)
    split_size = math.ceil(flattened.numel() / num_splits)
    return torch.split(flattened, split_size, dim=0)


def _gather_splits_direct_to_output(splits, output_tensor, input_numel, replica_groups, world_size):
    """Gather splits from all ranks directly into output tensor.

    Args:
        splits: Tuple of tensor splits to gather
        output_tensor: Pre-allocated output tensor to write results
        input_numel: Number of elements in original input tensor
        replica_groups: Replica groups for collective operation
        world_size: Number of ranks in the process group
    """
    from .ops import all_gather_op

    if not splits:
        return

    output_flat = output_tensor.view(-1)

    split_offset = 0
    for split in splits:
        split_size = split.numel()

        # Create views into output tensor at the correct positions for each rank
        rank_views = []
        for rank_idx in range(world_size):
            output_offset = rank_idx * input_numel + split_offset
            # View points directly to the target location in output_tensor
            view = output_flat[output_offset : output_offset + split_size].view(split.shape)
            rank_views.append(view)

        # Gather directly into the views - no copy needed!
        all_gather_op(split, replica_groups, out=tuple(rank_views), slice_output=True)

        split_offset += split_size


def _gather_splits_with_rank_views(splits, output_tensors_list, replica_groups, world_size):
    """Gather splits into rank-sliced output tensors using views.

    Args:
        splits: Tuple of tensor splits to gather
        output_tensors_list: List of output tensors (one per rank)
        replica_groups: Replica groups for collective operation
        world_size: Number of ranks in the process group
    """
    from .ops import all_gather_op

    split_offset = 0
    for split in splits:
        split_size = split.numel()

        rank_views = []
        for rank_idx in range(world_size):
            output_flat = output_tensors_list[rank_idx].view(-1)
            view = output_flat[split_offset : split_offset + split_size].view(split.shape)
            rank_views.append(view)

        all_gather_op(split, replica_groups, out=tuple(rank_views), slice_output=True)
        split_offset += split_size


def _should_use_split_allgather(tensor, world_size, bucket_size_bytes):
    """Check if split-based allgather should be used based on output size.

    Args:
        tensor: Input tensor to check
        world_size: Number of ranks in the process group
        bucket_size_bytes: Bucket size threshold in bytes

    Returns:
        True if split-based allgather should be used, False otherwise
    """
    output_bytes = tensor.numel() * tensor.element_size() * world_size
    return output_bytes > bucket_size_bytes


def _validate_no_zero_dim0(tensors):
    """Validate tensors don't have zero size in dimension 0."""
    flat_tensors = _flatten_tensors(tensors)

    empty_tensors = [
        (i, tensor.shape) for i, tensor in enumerate(flat_tensors) if tensor.size(0) == 0
    ]
    if empty_tensors:
        tensor_info = ", ".join([f"tensor[{i}]: {shape}" for i, shape in empty_tensors])
        raise ValueError(f"tensors cannot have zero size in dimension 0, found: {tensor_info}")


def _execute_with_xla_op_check(xla_op, op_name, core_fn, fallback_fn, args, kwargs):
    """Execute operation with XLA op can_handle check.

    Args:
        xla_op: XLA operation instance with can_handle method
        op_name: Name of the operation for error messages
        core_fn: callable if can_handle returns True
        fallback_fn: callable to execute if can_handle returns False
        args: Positional arguments for error reporting
        kwargs: Keyword arguments for error reporting
    """
    can_handle_result = xla_op.can_handle(*args, **kwargs)
    if isinstance(can_handle_result, tuple):
        can_handle_impl, can_handle_error_msg = can_handle_result
    else:
        can_handle_impl, can_handle_error_msg = can_handle_result, None

    if can_handle_impl:
        core_fn()
    else:
        base_debug_msg = f"No implementation could handle operation {op_name}. "
        _create_and_raise_detailed_error(
            RuntimeError, op_name, base_debug_msg + can_handle_error_msg, args, kwargs, None
        )


def _validate_reduce_scatter_inputs(input_tensor, output, world_size):
    """Validate inputs for reduce-scatter operation."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if input_tensor.numel() == 0:
        raise ValueError("input_tensor cannot be empty")
    if output.numel() == 0:
        raise ValueError("output tensor cannot be empty")


def _find_scatter_dim(input_tensor, output):
    """Find scatter dimension, handling rank mismatches."""
    input_shape = list(input_tensor.shape)
    output_shape = list(output.shape)

    if len(input_shape) > len(output_shape):
        output_shape = [1] * (len(input_shape) - len(output_shape)) + output_shape
    elif len(output_shape) > len(input_shape):
        input_shape = [1] * (len(output_shape) - len(input_shape)) + input_shape

    diff_indices = [
        i for i, (a, b) in enumerate(zip(input_shape, output_shape, strict=False)) if a != b
    ]
    return diff_indices[0] if diff_indices else 0


def _calculate_bucket_size(input_tensor, world_size):
    """Calculate max elements per bucket for reduce-scatter."""
    element_size = input_tensor.element_size()
    if element_size == 0:
        raise ValueError("input_tensor has invalid element_size of 0")

    min_buckets = int(os.environ.get("COLLECTIVE_MIN_BUCKETS", "0"))

    if input_tensor.numel() * element_size < _COLLECTIVE_BUCKETSIZE_BYTES:
        if min_buckets > 1 and input_tensor.numel() > min_buckets:
            return max(world_size, input_tensor.numel() // min_buckets)
        else:
            return input_tensor.numel()
    else:
        return _COLLECTIVE_BUCKETSIZE_BYTES // element_size


def _flatten_input_for_scatter(input_tensor, scatter_dim, world_size, input_breaker):
    """Flatten input tensor optimized for scatter dimension."""
    if scatter_dim == 0:
        return input_tensor.view(-1)

    shape = list(input_tensor.shape)
    shape[scatter_dim : scatter_dim + 1] = [world_size, input_breaker]
    reshaped = input_tensor.view(shape)

    perm = [
        scatter_dim,
        scatter_dim + 1,
        *list(range(scatter_dim)),
        *list(range(scatter_dim + 2, len(shape))),
    ]
    transposed = reshaped.permute(perm).contiguous()
    return transposed.view(-1)


def _calculate_chunking_params(num_elements_per_block, num_elem_to_pick_per_block):
    """Calculate number of chunks and remainder for bucketing."""
    if num_elem_to_pick_per_block > 0:
        num_chunks = num_elements_per_block // num_elem_to_pick_per_block
        remainder = num_elements_per_block % num_elem_to_pick_per_block
    else:
        num_chunks = 0
        remainder = 0
    return num_chunks, remainder


def _create_input_bucket_generator(
    input_flat,
    world_size,
    num_elements_per_block,
    num_elem_to_pick_per_block,
    num_chunks,
    remainder,
):
    """Create generator for input buckets."""
    if num_chunks > 0:
        for chunk_idx in range(num_chunks):
            yield torch.cat(
                [
                    input_flat[
                        rank_idx * num_elements_per_block
                        + chunk_idx * num_elem_to_pick_per_block : rank_idx * num_elements_per_block
                        + chunk_idx * num_elem_to_pick_per_block
                        + num_elem_to_pick_per_block
                    ]
                    for rank_idx in range(world_size)
                ],
                dim=0,
            )

        if remainder > 0:
            yield torch.cat(
                [
                    input_flat[
                        rank_idx * num_elements_per_block
                        + num_chunks * num_elem_to_pick_per_block : rank_idx
                        * num_elements_per_block
                        + num_chunks * num_elem_to_pick_per_block
                        + remainder
                    ]
                    for rank_idx in range(world_size)
                ],
                dim=0,
            )
    else:
        yield input_flat


def _create_output_bucket_generator(output, num_elem_to_pick_per_block, num_chunks, remainder):
    """Create generator for output bucket views."""
    output_flat = output.view(-1)
    offset = 0
    total_buckets = (num_chunks + (1 if remainder > 0 else 0)) if num_chunks > 0 else 1

    for bucket_idx in range(total_buckets):
        if bucket_idx < num_chunks:
            chunk_size = num_elem_to_pick_per_block
        elif bucket_idx == num_chunks and remainder > 0:
            chunk_size = remainder
        else:
            chunk_size = output.numel()

        yield output_flat.narrow(0, offset, chunk_size)
        offset += chunk_size


def get_reduce_scatter_inputs_outputs(input_tensor, output, world_size):
    """Split input tensor for bucketed reduce-scatter operation.

    Args:
        input_tensor: Input tensor to split
        output: Expected output tensor
        world_size: Number of ranks in the process group

    Returns:
        Tuple of (input_generator, output_generator)
    """
    _validate_reduce_scatter_inputs(input_tensor, output, world_size)

    scatter_dim = _find_scatter_dim(input_tensor, output)
    max_elements_input_chunk_rs = _calculate_bucket_size(input_tensor, world_size)

    num_elements_per_block = input_tensor.numel() // world_size
    num_elem_to_pick_per_block = max_elements_input_chunk_rs // world_size
    input_breaker = input_tensor.size(scatter_dim) // world_size

    input_flat = _flatten_input_for_scatter(input_tensor, scatter_dim, world_size, input_breaker)
    num_chunks, remainder = _calculate_chunking_params(
        num_elements_per_block, num_elem_to_pick_per_block
    )

    input_gen = _create_input_bucket_generator(
        input_flat,
        world_size,
        num_elements_per_block,
        num_elem_to_pick_per_block,
        num_chunks,
        remainder,
    )
    output_gen = _create_output_bucket_generator(
        output, num_elem_to_pick_per_block, num_chunks, remainder
    )

    return input_gen, output_gen


def reconstruct_reduce_scatter_output(output_tensor_cat, output, input_breaker, scatter_dim):
    """Reconstruct output tensor from bucketed reduce-scatter results.

    Args:
        output_tensor_cat: Concatenated output from bucketed operations
        output: Target output tensor
        input_breaker: Size of each chunk along scatter dimension
        scatter_dim: Dimension along which scattering occurred

    Returns:
        List containing the reconstructed output tensor

    Raises:
        ValueError: If output_tensor_cat size doesn't match expected output size
    """
    # Validate input sizes
    expected_size = output.numel()
    if output_tensor_cat.numel() != expected_size:
        raise ValueError(
            f"Size mismatch in reconstruct_reduce_scatter_output: "
            f"expected {expected_size} elements, got {output_tensor_cat.numel()}"
        )

    flat_output = output_tensor_cat

    # Calculate world_size from output and input_breaker
    world_size = output.size(scatter_dim) // input_breaker

    # Compute shape for reshaping: reverse the flatten->permute->reshape from forward pass
    shape_before_flatten = list(output.shape)
    shape_before_flatten[scatter_dim : scatter_dim + 1] = [world_size, input_breaker]

    # Reshape flat_output to [world_size, input_breaker, ...other dims...]
    perm = [
        scatter_dim,
        scatter_dim + 1,
        *list(range(scatter_dim)),
        *list(range(scatter_dim + 2, len(shape_before_flatten))),
    ]
    temp_shape = [shape_before_flatten[i] for i in perm]
    reshaped = flat_output.reshape(temp_shape)

    # Inverse permute to get back to [..., world_size, input_breaker, ...]
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    transposed = reshaped.permute(inv_perm)

    # Merge world_size and input_breaker dimensions back
    final_shape = list(output.shape)
    reconstructed = transposed.reshape(final_shape)

    output.copy_(reconstructed)
    return [output]
