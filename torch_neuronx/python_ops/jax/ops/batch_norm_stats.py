"""JAX implementation of batch_norm_stats operation."""

import jax.lax as lax
import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(
    ["aten::batch_norm_stats", "aten::batch_norm_stats.out"],
    static_argnums=(1,),
    static_argnames=("eps",),
)
def _aten_batch_norm_stats(input, eps):
    """Compute batch normalization statistics (mean and inverse std).

    Args:
        input: Input tensor with shape [N, C, ...]
        eps: Epsilon for numerical stability

    Returns:
        mean: Channel-wise mean [C]
        invstd: Channel-wise inverse standard deviation [C]
    """
    # Determine reduction axes: all axes except the channel dimension (index 1)
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)

    # Calculate mean across the determined reduction_axes, NOT keeping dimensions
    computed_mean = jnp.mean(input, axis=reduction_axes, keepdims=False)
    # Calculate variance, NOT keeping dimensions
    var = jnp.var(input, axis=reduction_axes, ddof=0, keepdims=False)
    # Calculate inverse standard deviation using jax.lax.rsqrt
    computed_invstd = lax.rsqrt(var + eps)

    return computed_mean, computed_invstd
