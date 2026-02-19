"""Test that unary operations backward pass is properly registered
with PyTorch dispatcher."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops
from torch_neuronx.utils import use_mlir_aten_ops


# Define wrapper functions for ops that need additional parameters
def softmax_wrapper(x):
    return F.softmax(x, dim=0)


def log_softmax_wrapper(x):
    return F.log_softmax(x, dim=0)


def pow_wrapper(x):
    # Use a constant exponent of 2 (square)
    return torch.pow(x, 2.0)


# List of unary operations to test with their corresponding backward ops
UNARY_OPS = [
    # Elementwise unary ops: (op_func, op_name, backward_op_name)
    (torch.log, "aten::log", None),
    (torch.sigmoid, "aten::sigmoid", "aten::sigmoid_backward"),
    (torch.sin, "aten::sin", None),
    (torch.cos, "aten::cos", None),
    (F.silu, "aten::silu", "aten::silu_backward"),
    (torch.exp, "aten::exp", None),
    (torch.abs, "aten::abs", None),
    (torch.relu, "aten::relu", None),
    (F.gelu, "aten::gelu", "aten::gelu_backward"),
    (torch.sqrt, "aten::sqrt", None),
    (torch.neg, "aten::neg", None),
    (F.softplus, "aten::softplus", "aten::softplus_backward"),
    # Wrappers for unary ops that need extra arguments
    (softmax_wrapper, "aten::softmax", "aten::_softmax_backward_data"),
    (log_softmax_wrapper, "aten::log_softmax", "aten::_log_softmax_backward_data"),
    (pow_wrapper, "aten::pow", None),
]

# Define operations that are expected to fail with specific reasons
# Add operations and their failure reasons to this dictionary
XFAIL_OPS = {
    # "op_name": "Reason for failure",
}
if not use_mlir_aten_ops():
    XFAIL_OPS["aten::softplus"] = "aten::softplus only supported using dynamo decompositions"
    XFAIL_OPS["aten::gelu"] = "aten::gelu only supported using dynamo decompositions"


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestUnaryOpsBackward:
    """Test unary operations backward pass."""

    @pytest.mark.parametrize(
        "op_func, op_name, backward_op_name",
        [
            pytest.param(
                op_func,
                op_name,
                backward_op_name,
                marks=pytest.mark.xfail(reason=XFAIL_OPS[op_name]),
            )
            if op_name in XFAIL_OPS
            else (op_func, op_name, backward_op_name)
            for op_func, op_name, backward_op_name in UNARY_OPS
        ],
    )
    def test_unary_backward_sanity(self, op_func, op_name, backward_op_name):
        """Test basic unary backward operation."""
        # Create input tensor that requires grad
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float32)

        x_neuron = x.detach().clone().to("neuron")
        x_neuron.requires_grad = True

        with track_neuron_ops():
            # Forward pass
            y = op_func(x)
            y_neuron = op_func(x_neuron)

            # Verify requires_grad is preserved
            assert y_neuron.requires_grad

            # Create random gradient tensor between 0 and 1 with same shape as x
            grad_output = torch.rand_like(x)
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            # Verify operations ran on Neuron
            assert_op_runs_on_neuron(f"aten::{op_name}")
            if backward_op_name:
                assert_op_runs_on_neuron(backward_op_name)

        # Check that gradients match
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)
