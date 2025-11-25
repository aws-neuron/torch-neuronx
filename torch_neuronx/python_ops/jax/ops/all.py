# """XLA implementation of all/reduction operation."""

import jax.numpy as jnp

from torch_neuronx.python_ops.jax.jax_impl import _with_reduction_scalar
from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(
    ["aten::all", "aten::all.out", "aten::all.all_out"],
    operation_type="reduction",
    static_argnums=(1, 2),
)
def _aten_all(self, dim=None, keepdim=False):
    # dtype parameter is passed by PyTorch but not used by JAX's any
    return _with_reduction_scalar(jnp.all, self, dim, keepdim)
