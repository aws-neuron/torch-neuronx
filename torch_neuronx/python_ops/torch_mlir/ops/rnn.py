"""RNN operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::_thnn_fused_lstm_cell"])
def torch_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    """Run the fused LSTM cell through the Neuron decomposition.

    Args:
        input_gates (torch.Tensor): Input-projection gate preactivations.
        hidden_gates (torch.Tensor): Hidden-projection gate preactivations.
        cx (torch.Tensor): Previous cell state.
        input_bias (torch.Tensor, optional): Input-gate bias. Defaults to ``None``.
        hidden_bias (torch.Tensor, optional): Hidden-gate bias. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden state, cell state, and
            activated-gate workspace.
    """
    return torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )


@register_aten(["aten::_thnn_fused_lstm_cell_backward_impl"], static_argnums=(5,))
def torch_thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    """Run the fused LSTM cell backward through the Neuron decomposition.

    Args:
        grad_hy (torch.Tensor or None): Hidden-state gradient, or ``None``.
        grad_cy (torch.Tensor or None): Cell-state gradient, or ``None``.
        cx (torch.Tensor): Previous cell state.
        cy (torch.Tensor): Current cell state.
        workspace (torch.Tensor): Activated gates saved by the forward pass.
        has_bias (bool): Whether to compute a combined gate-bias gradient.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]: Combined
            gate-preactivation, previous-cell, and optional combined bias gradients; all are
            ``None`` when both input gradients are ``None``.
    """
    return torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )
