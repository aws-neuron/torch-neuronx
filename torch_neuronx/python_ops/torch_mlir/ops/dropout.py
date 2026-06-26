"""Dropout operations."""

import torch

from torch_neuronx.python_ops.nki_kernels.rng_state import _has_nki_rng

from ..operation_registry import register_aten


@register_aten(
    ["aten::native_dropout", "aten::native_dropout.out"],
    static_argnums=(1, 2),
    output_params=("out0", "out1"),
)
def torch_native_dropout(input, p: float, train: bool = True, out0=None, out1=None, **kwargs):
    """Decompose native_dropout using on-device RNG."""
    if train and not (0.0 <= float(p) <= 1.0):
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not train or p == 0.0:
        mask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        scale = 1.0
    elif p == 1.0:
        mask = torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        scale = 0.0
    else:
        keep_prob = 1.0 - p
        if _has_nki_rng:
            mask = torch.ops.nki_kernels.rand(input, list(input.shape)) < keep_prob
        else:
            mask = (
                torch.rand(list(input.shape), dtype=torch.float32, device=input.device) < keep_prob
            )
        scale = 1.0 / keep_prob

    y = input * mask.to(input.dtype) * scale
    return y, mask


@register_aten(["aten::native_dropout_backward"], static_argnums=(2,))
def torch_native_dropout_backward(grad_output, mask, scale: float):
    """Backward for dropout: grad_input = grad_output * mask * scale."""
    return torch.ops.aten.native_dropout_backward(grad_output, mask, scale)
