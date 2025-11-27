"""JAX implementation of batch_norm_backward_reduce operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(["aten::batch_norm_backward_reduce", "aten::batch_norm_backward_reduce.out"])
def _aten_batch_norm_backward_reduce(
    grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g
):
    """Backward pass reduction for batch normalization.

    Args:
        input_g, weight_g, bias_g: Gradient flags (unused - always compute all gradients)
    """
    # Reduction axes: all except channel dimension (index 1)
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)

    # Reshape statistics for broadcasting
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = jnp.reshape(mean, shape)
    invstd_reshaped = jnp.reshape(invstd, shape)

    # Compute normalized input
    xmu = input - mean_reshaped
    xhat = xmu * invstd_reshaped

    # Compute sum_dy and sum_dy_xmu
    sum_dy = jnp.sum(grad_output, axis=reduction_axes, keepdims=False)
    sum_dy_xmu = jnp.sum(grad_output * xmu, axis=reduction_axes, keepdims=False)

    # Compute gradients (always compute, return based on flags)
    grad_weight = jnp.sum(grad_output * xhat, axis=reduction_axes, keepdims=False)
    grad_bias = sum_dy

    return sum_dy, sum_dy_xmu, grad_weight, grad_bias
