"""JAX implementation for torch.histc."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


def safe_histogram(data, bins, range_):
    """Fast histogram using linear interpolation like PyTorch"""
    data = jnp.ravel(data)

    # Linear interpolation (same as PyTorch)
    bin_idx = ((data - range_[0]) * bins / (range_[1] - range_[0])).astype(jnp.int32)

    # Clamp to valid range and handle edge cases
    in_range = (data >= range_[0]) & (data <= range_[1])
    bin_idx = jnp.where(bin_idx >= bins, bins - 1, bin_idx)
    bin_idx = jnp.where(bin_idx < 0, 0, bin_idx)

    # Neuron-optimized: vectorized counting without scatter
    bin_indices = jnp.arange(bins)
    matches = bin_idx[:, None] == bin_indices[None, :]  # Shape: (N, bins)
    valid_matches = matches & in_range[:, None]
    counts = jnp.sum(valid_matches, axis=0).astype(data.dtype)

    return counts


@register_aten(
    [
        "aten::histc",
        "aten::histc.out",
    ],
    static_argnums=(1, 2, 3),
    static_argnames=("__out_dtype",),
)
def _aten_histc(input, bins=100, min=0, max=0, out=None, __out_dtype=None):
    """JAX implementation for torch.histc operation.

    This implementation does NOT use jnp.histogram directly as its HLO includes index out-of-range.
    For more details, refer to the CR description.
    """

    if __out_dtype is not None and input.dtype != __out_dtype:
        raise TypeError(
            "input tensor and hist tensor should have the same dtype, "
            f"but got input {input.dtype} and hist {__out_dtype}"
        )

    if min == 0 and max == 0 and isinstance(input, jnp.ndarray) and input.size != 0:
        min = jnp.min(input)
        max = jnp.max(input)

    # Handle special case where min == max
    # Reference: https://github.com/jax-ml/jax/blob/be26e454940dacac0ee471ecc4c45b5ccf8984fd/jax/_src/numpy/lax_numpy.py#L794C1-L795C72
    range_ = jnp.asarray((min, max))
    range_ = (
        jnp.where(jnp.ptp(range_) == 0, range_[0] - 0.5, range_[0]),
        jnp.where(jnp.ptp(range_) == 0, range_[1] + 0.5, range_[1]),
    )
    hist = safe_histogram(input, bins, range_)
    return hist
