"""RNN operations."""

import torch

from ..operation_registry import register_aten


@register_aten(["aten::_thnn_fused_lstm_cell"])
def torch_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    """Fused LSTM cell forward; decomposed via the Neuron decomposition table."""
    return torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )


@register_aten(["aten::_thnn_fused_lstm_cell_backward_impl"], static_argnums=(5,))
def torch_thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    """Fused LSTM cell backward; decomposed via the Neuron decomposition table."""
    return torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, has_bias
    )
