import jax
import numpy as np
import torch

from torch_neuronx.python_ops.jax.operation_registry import register_aten


@register_aten(
    "aten::one_hot",
    static_argnums=(1,),
    uses_preprocessing=True,
)
def _aten_one_hot(x, num_classes=-1):
    # Check for negative indices (matching PyTorch's behavior)
    min_index = torch.min(x).item()
    if min_index < 0:
        raise RuntimeError("Class values must be non-negative.")

    # If num_classes is -1, infer it from data
    if num_classes < 0:
        num_classes = torch.max(x).item() + 1

    # Check for out-of-range indices (matching PyTorch's behavior)
    max_index = torch.max(x).item()
    if max_index >= num_classes:
        raise RuntimeError("Class values must be smaller than num_classes.")

    def _one_hot_fn(x, num_classes):
        return jax.nn.one_hot(x, num_classes, dtype=np.int32)

    return _one_hot_fn, (x, num_classes), {}
