"""JAX implementations for addcmul and addcdiv using function registration."""

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten("aten::addcmul.out")
def _aten_addcmul(input, tensor1, tensor2, *, value=1.0, out=None):
    """Compute input + value * (tensor1 * tensor2)."""
    return input + (value * (tensor1 * tensor2))


@register_aten("aten::addcdiv.out")
def _aten_addcdiv(input, tensor1, tensor2, *, value=1.0, out=None):
    """Compute input + value * (tensor1 / tensor2)."""
    return input + (value * (tensor1 / tensor2))
