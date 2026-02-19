"""Loss function operations."""

import torch
from torch._refs.nn.functional import TensorLikeType

from ..operation_registry import register_aten


@register_aten(["aten::nll_loss_forward"], static_argnums=(3, 4))
def nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None,
    reduction: int,
    ignore_index: int,
):
    """NLL loss forward using PyTorch decomposition."""
    if input.dim() == 1 and target.dim() == 1 and target.size(0) != 1:
        raise ValueError("For 1D input, 1D target must have size 1")
    return torch.ops.aten.nll_loss_forward(input, target, weight, reduction, ignore_index)


@register_aten(["aten::nll_loss_backward"], static_argnums=(4, 5))
def nll_loss_backward(
    grad_output: TensorLikeType,
    input: TensorLikeType,
    target: TensorLikeType,
    weight: TensorLikeType | None,
    reduction: int,
    ignore_index: int,
    total_weight: TensorLikeType,
):
    """NLL loss backward using PyTorch decomposition."""
    return torch.ops.aten.nll_loss_backward(
        grad_output, input, target, weight, reduction, ignore_index, total_weight
    )
