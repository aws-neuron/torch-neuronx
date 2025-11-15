"""JAX implementation for torch.histc."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


def safe_histogram(data, bins, range_):
    """Safe histogram implementation using digitize + bincount"""
    # bincount only supports 1-D array
    data = jnp.ravel(data)

    # Create bin edges
    bin_edges = jnp.linspace(range_[0], range_[1], bins + 1)

    # Find bin indices (right=False means left edge is inclusive)
    # digitize returns:
    #   0 for values < range_[0]
    #   1 to bins for values in range
    #   bins+1 for values > range_[1]
    bin_indices = jnp.digitize(data, bin_edges, right=False)

    # Special case: values exactly at upper bound should be in last bin
    # digitize returns bins+1 for value >= range_[1], but we want bins for value == range_[1]
    at_upper_bound = data == range_[1]
    bin_indices = jnp.where(at_upper_bound, bins, bin_indices)

    # Create mask for valid range: [1, bins]
    # 0 means < min, bins+1 means > max (both invalid)
    in_range = (bin_indices >= 1) & (bin_indices <= bins)

    # Convert to 0-indexed bins
    bin_indices = bin_indices - 1

    # Clamp to valid range [0, bins-1] to prevent bincount errors
    bin_indices = jnp.clip(bin_indices, 0, bins - 1)

    # Create weights: 1.0 for in-range values, 0.0 for out-of-range
    weights = jnp.where(in_range, 1.0, 0.0)
    counts = jnp.bincount(bin_indices, weights=weights, length=bins)

    return counts, bin_edges


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
    hist, bin_edges = safe_histogram(input, bins, range_)
    return hist.astype(input.dtype)
