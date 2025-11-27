"""JAX implementation of batch_norm_backward_elemt operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(["aten::batch_norm_backward_elemt", "aten::batch_norm_backward_elemt.out"])
def _aten_batch_norm_backward_elemt(
    grad_output, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
):
    """Element-wise backward pass for batch normalization.

    Args:
        grad_output: Gradient w.r.t. output
        input: Original input tensor
        mean: Channel-wise mean
        invstd: Channel-wise inverse standard deviation
        weight: Scale parameter (can be None)
        sum_dy: Sum of gradients (from reduce step)
        sum_dy_xmu: Sum of grad * (input - mean) (from reduce step)
        count: Total element count for normalization

    Returns:
        grad_input: Gradient w.r.t. input
    """
    # Reshape statistics for broadcasting
    shape = [1] * input.ndim
    shape[1] = -1

    mean_reshaped = jnp.reshape(mean, shape)
    invstd_reshaped = jnp.reshape(invstd, shape)
    weight_reshaped = jnp.reshape(weight, shape) if weight is not None else 1.0
    sum_dy_reshaped = jnp.reshape(sum_dy, shape)
    sum_dy_xmu_reshaped = jnp.reshape(sum_dy_xmu, shape)

    # Compute total count for normalization
    total_count = jnp.sum(count)

    # Standard batch norm backward formula
    xmu = input - mean_reshaped
    k = sum_dy_xmu_reshaped * invstd_reshaped * invstd_reshaped / total_count
    grad_input = (
        (grad_output - sum_dy_reshaped / total_count - xmu * k) * invstd_reshaped * weight_reshaped
    )

    return grad_input
