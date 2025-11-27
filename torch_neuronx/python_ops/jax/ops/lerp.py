import jax.numpy as jnp

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(
    [
        "aten::lerp",
        "aten::lerp.Scalar",
        "aten::lerp.Tensor",
        "aten::lerp_",
        "aten::lerp_.Scalar",
        "aten::lerp_.Tensor",
    ],
)
def _aten_lerp(input, end, weight):
    """JAX implementation of torch.lerp operation.
    Reference - https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py#L5286"""

    mask = jnp.abs(weight) >= 0.5
    coeff = jnp.where(mask, weight - 1.0, weight)
    base = jnp.where(mask, end, input)
    return coeff * (end - input) + base
