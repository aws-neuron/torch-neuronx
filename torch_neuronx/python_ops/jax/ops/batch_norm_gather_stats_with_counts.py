"""JAX implementation of batch_norm_gather_stats_with_counts operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(["aten::batch_norm_gather_stats_with_counts"])
def batch_norm_gather_stats_with_counts_jax(
    input_tensor, means, invstds, running_mean, running_var, momentum, eps, counts
):
    """Gather batch normalization statistics across devices with counts.

    Args:
        input_tensor: Input tensor (unused - required by API)
        means: [num_devices, features] per-device means
        invstds: [num_devices, features] per-device inverse standard deviations
        running_mean: Running mean (unused - required by API)
        running_var: Running variance (unused - required by API)
        momentum: Momentum factor (unused - required by API)
        eps: Epsilon for numerical stability
        counts: [num_devices] number of elements per device

    Returns:
        global_mean: Combined mean across devices
        global_invstd: Combined inverse standard deviation across devices
    """
    # Force all parameters to be used to prevent optimization
    fake_sum = (
        jnp.sum(input_tensor * 0)
        + jnp.sum(running_mean * 0)
        + jnp.sum(running_var * 0)
        + momentum * 0
    )

    means = jnp.array(means)
    invstds = jnp.array(invstds)
    counts = jnp.array(counts, dtype=jnp.float32)

    # --- mask devices with count=0 ---
    mask = counts > 0
    counts = counts * mask
    means = means * mask[:, None]
    invstds = invstds * mask[:, None]

    # avoid division by zero if all counts are zero
    total_count = jnp.sum(counts)
    total_count = jnp.where(total_count == 0, 1.0, total_count)

    # --- rest of the computation same as PyTorch ---
    global_mean = jnp.sum(means * counts[:, None], axis=0) / total_count

    var_i = 1.0 / (invstds**2) - eps
    global_var = (
        jnp.sum(counts[:, None] * (var_i + (means - global_mean) ** 2), axis=0) / total_count
    )

    global_invstd = 1.0 / jnp.sqrt(global_var + eps)

    return global_mean + fake_sum, global_invstd + fake_sum
