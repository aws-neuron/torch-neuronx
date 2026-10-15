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
    (F.logsigmoid, "aten::log_sigmoid_forward", "aten::log_sigmoid_backward"),
    (F.elu, "aten::elu", "aten::elu_backward"),
    (F.hardtanh, "aten::hardtanh", "aten::hardtanh_backward"),
    (F.hardshrink, "aten::hardshrink", "aten::hardshrink_backward"),
    (F.hardswish, "aten::hardswish", "aten::hardswish_backward"),
    (F.mish, "aten::mish", "aten::mish_backward"),
    # selu and relu6 are CompositeImplicitAutograd. They decompose to elu and
    # hardtanh, so the backward runs as elu_backward and hardtanh_backward.
    (torch.selu, "aten::elu", "aten::elu_backward"),
    (F.relu6, "aten::hardtanh", "aten::hardtanh_backward"),
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

    def test_abs_backward_negative(self):
        """abs backward returns sign(x) * grad_output; exercise negative inputs."""
        x = torch.tensor([-3.0, -1.0, 2.0, 4.0], requires_grad=True, dtype=torch.float32)
        x_neuron = x.detach().clone().to("neuron")
        x_neuron.requires_grad = True

        with track_neuron_ops():
            y = torch.abs(x)
            y_neuron = torch.abs(x_neuron)

            grad_output = torch.rand_like(x)
            y.backward(grad_output)
            y_neuron.backward(grad_output.to("neuron"))

            assert_op_runs_on_neuron("aten::abs")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_elu_selu_forward_reference_accuracy(self):
        # elu's negative branch is scale*alpha*expm1(input_scale*x). The hardware
        # exp misses the strict reference-numerics oracle, so the lowering uses the
        # accurate expm1. Check elu and selu against CPU at the oracle tolerance.
        x = torch.linspace(-20.0, 20.0, 4000, dtype=torch.float32)
        x_neuron = x.to("neuron")
        for fn in (torch.selu, F.elu):
            got = fn(x_neuron).cpu()
            torch.testing.assert_close(got, fn(x), rtol=1.3e-6, atol=1e-5)


class TestGradInputVariants:
    """Tests for .grad_input variant dispatch to Neuron."""

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_gelu_backward_grad_input_variant(self, shape):
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        x = torch.randn(shape, device="neuron")
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.gelu_backward.grad_input(
                grad_output, x, approximate="none", grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::gelu_backward")
        expected = torch.ops.aten.gelu_backward(grad_output, x, approximate="none")
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_softplus_backward_grad_input_variant(self, shape):
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        x = torch.randn(shape, device="neuron")
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.softplus_backward.grad_input(
                grad_output, x, 1.0, 20.0, grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::softplus_backward")
        expected = torch.ops.aten.softplus_backward(grad_output, x, 1.0, 20.0)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_log_sigmoid_backward_grad_input_variant(self, shape):
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        x = torch.randn(shape, device="neuron")
        fwd_result = torch.ops.aten.log_sigmoid_forward(x)
        buffer = fwd_result[1]
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.log_sigmoid_backward.grad_input(
                grad_output, x, buffer, grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::log_sigmoid_backward")
        expected = torch.ops.aten.log_sigmoid_backward(grad_output, x, buffer)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_elu_backward_grad_input_variant(self, shape):
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        self_or_result = torch.randn(shape, device="neuron")
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.elu_backward.grad_input(
                grad_output, 1.5, 1.0, 1.0, False, self_or_result, grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::elu_backward")
        expected = torch.ops.aten.elu_backward(grad_output, 1.5, 1.0, 1.0, False, self_or_result)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_hardtanh_backward_grad_input_variant(self, shape):
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        x = torch.randn(shape, device="neuron")
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.hardtanh_backward.grad_input(
                grad_output, x, 0.0, 6.0, grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::hardtanh_backward")
        expected = torch.ops.aten.hardtanh_backward(grad_output, x, 0.0, 6.0)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    @pytest.mark.parametrize("shape", [(32,), (8, 16)])
    def test_hardshrink_backward_grad_input_variant(self, shape):
        torch.manual_seed(42)
        grad_output = torch.randn(shape, device="neuron")
        x = torch.randn(shape, device="neuron")
        grad_input = torch.empty(shape, device="neuron")

        with track_neuron_ops():
            result = torch.ops.aten.hardshrink_backward.grad_input(
                grad_output, x, 0.5, grad_input=grad_input
            )
        assert_op_runs_on_neuron("aten::hardshrink_backward")
        expected = torch.ops.aten.hardshrink_backward(grad_output, x, 0.5)
        torch.testing.assert_close(result.cpu(), expected.cpu())
        torch.testing.assert_close(grad_input.cpu(), expected.cpu())

    def test_elu_inplace_backward_negative_alpha_raises(self):
        # The in-place elu_ (is_result=True) with a negative alpha cannot recover
        # the gradient from its output. aten raises; the Neuron lowering must too.
        a = torch.tensor([-1.0, 1.0], device="neuron", requires_grad=True)
        out = torch.nn.functional.elu_(a.clone(), alpha=-2.0)
        with pytest.raises(RuntimeError, match="call out-of-place version"):
            out.backward(torch.ones(2, device="neuron"))
