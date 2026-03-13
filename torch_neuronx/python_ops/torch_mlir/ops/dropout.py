"""Dropout operations."""

import torch

from ..operation_registry import register_aten


@register_aten(
    ["aten::native_dropout"],
    static_argnums=(1, 2),
    uses_preprocessing=True,
)
def torch_native_dropout(input, p: float, train: bool = True, **kwargs):
    """Native dropout implementation with CPU preprocessing for random mask generation."""

    def torch_dropout_fn(x, p, train, mask, scale):
        y = x * mask.to(x.dtype) * scale
        return y, mask

    if train and not (0.0 <= float(p) <= 1.0):
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not train or p == 0.0:
        mask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        scale = 1.0
    elif p == 1.0:
        mask = torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        scale = 0.0
    else:
        # Generate mask on CPU then move to device
        mask = torch.empty(input.shape, dtype=torch.bool)
        keep_prob = 1.0 - p
        mask = torch.bernoulli(mask, keep_prob).to(input.device)
        scale = 1.0 / keep_prob

    return torch_dropout_fn, (input, p, train, mask, scale), {}


@register_aten(["aten::native_dropout_backward"], static_argnums=(2,))
def torch_native_dropout_backward(grad_output, mask, scale: float):
    """Backward for dropout: grad_input = grad_output * mask * scale."""
    return torch.ops.aten.native_dropout_backward(grad_output, mask, scale)
