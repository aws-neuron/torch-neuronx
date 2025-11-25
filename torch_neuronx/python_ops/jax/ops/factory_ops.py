"""Factory operations for tensor creation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten

from ..type_converter import convert_dtype_with_default


def _ones(size, dtype=None):
    """Create a tensor filled with ones."""
    return jnp.ones(size, dtype)


@register_aten(
    [
        "aten::ones",
        "aten::ones.out",
    ],
    static_argnums=(0,),
    static_argnames=("dtype",),
)
def _aten_ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    """JAX implementation of torch.ones operation."""
    jdtype = convert_dtype_with_default(dtype)

    try:
        sz = tuple(size)
    except TypeError:
        sz = (size,)

    return _ones(sz, jdtype)
