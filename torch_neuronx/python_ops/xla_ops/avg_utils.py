"""Shared utilities for AVG operation handling in collective operations."""

import torch


def create_divisor_tensor(replica_groups, inputs):
    """Create divisor tensor for AVG operations.

    Args:
        replica_groups: List of replica groups
        inputs: Input tensors list

    Returns:
        torch.Tensor: Divisor tensor with appropriate dtype and device

    Raises:
        ValueError: If replica_groups is empty or inputs is empty
    """
    if not replica_groups or not inputs:
        raise ValueError("replica_groups and inputs cannot be empty")

    replica_group = replica_groups[0]
    num_replicas = len(replica_group)

    if num_replicas == 0:
        raise ValueError("Replica group cannot be empty")

    return torch.tensor(num_replicas, dtype=inputs[0].dtype, device=inputs[0].device)


def prepare_avg_inputs(inputs, reduce, replica_groups):
    """Prepare inputs for AVG operation by adding divisor if needed.

    Args:
        inputs: List of input tensors
        reduce: Reduction operation string
        replica_groups: List of replica groups

    Returns:
        list: Modified inputs list with divisor appended for AVG operations
    """
    if reduce == "AVG":
        divisor = create_divisor_tensor(replica_groups, inputs)
        return [*inputs, divisor]
    return inputs


def apply_avg_division_single(scribe, result, divisor_param, output_shape):
    """Apply division for AVG operation on single tensor result.

    Args:
        scribe: HLO scribe instance
        result: Result tensor from collective operation
        divisor_param: Divisor parameter tensor
        output_shape: Output shape for broadcasting

    Returns:
        HLO tensor: Result after division
    """
    broadcasted_divisor = output_shape.Broadcast(divisor_param, dimensions=[])
    return output_shape.Divide(result, broadcasted_divisor)


def apply_avg_division_tuple(scribe, result, divisor_param, output_shape_list):
    """Apply division for AVG operation on tuple result.

    Args:
        scribe: HLO scribe instance
        result: Tuple result from collective operation
        divisor_param: Divisor parameter tensor
        output_shape_list: List of output shapes

    Returns:
        HLO tuple: Result after division applied to each element
    """
    divided_results = []
    for i in range(len(output_shape_list)):
        element = output_shape_list[i].GetTupleElement(result, tuple_index=i)
        # Apply the same logic as the single case
        divided_result = apply_avg_division_single(
            scribe, element, divisor_param, output_shape_list[i]
        )
        divided_results.append(divided_result)

    # Create tuple using the scribe's tuple method with the shapes
    tuple_shape = scribe.tuple(*output_shape_list)
    return tuple_shape.Tuple(*divided_results)
