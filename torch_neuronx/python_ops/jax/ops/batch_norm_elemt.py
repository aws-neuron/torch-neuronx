"""JAX implementation of batch_norm_elemt operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(["aten::batch_norm_elemt", "aten::batch_norm_elemt.out"])
def _aten_batch_norm_elemt(input, weight, bias, mean, invstd, eps):
    """Element-wise batch normalization.

    Args:
        eps: Unused in this implementation (invstd already includes eps)
    """
    # Reshape statistics to broadcast with input (add dimensions for spatial dims)
    shape = [1] * input.ndim
    shape[1] = -1  # Channel dimension

    mean_reshaped = jnp.reshape(mean, shape)
    invstd_reshaped = jnp.reshape(invstd, shape)
    weight_reshaped = jnp.reshape(weight, shape) if weight is not None else 1.0
    bias_reshaped = jnp.reshape(bias, shape) if bias is not None else 0.0

    # Apply batch normalization: (input - mean) * invstd * weight + bias
    normalized = (input - mean_reshaped) * invstd_reshaped
    return normalized * weight_reshaped + bias_reshaped
