"""Test that threshold and threshold_backward operations
are properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops
from torch_neuronx.utils import use_mlir_aten_ops

# Test shapes: 1D, 2D, 3D, and larger tensors
SHAPES = [
    (8,),  # 1D basic
    (3, 4),  # 2D basic
    (2, 3, 4),  # 3D basic
    (10, 20),  # 2D larger
]

# Supported dtypes
DTYPES = [torch.float32, torch.float16]

# Threshold/value combinations to test
THRESHOLD_VALUE_PAIRS = [
    (0.0, 0.0),  # Basic ReLU-like behavior
    (0.5, 0.0),  # Positive threshold
    (-0.5, 0.0),  # Negative threshold
    (0.0, -1.0),  # Non-zero replacement value
    (0.1, 0.5),  # Both positive
]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.skipif(
    not use_mlir_aten_ops(), reason="aten::threshold only supported using dynamo decompositions"
)
class TestThresholdRegistration:
    """Test threshold operation registration and functionality."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("threshold,value", THRESHOLD_VALUE_PAIRS)
    def test_threshold_forward(self, shape, dtype, threshold, value):
        """Test threshold forward operation with various shapes, dtypes, and
        threshold/value pairs."""
        # Create input tensor with values around the threshold
        x = torch.randn(shape, dtype=dtype) * 2  # Scale to have values above/below threshold
        x_neuron = x.clone().to("neuron")

        with track_neuron_ops():
            # Forward pass on CPU
            y = torch.nn.functional.threshold(x, threshold, value)
            # Forward pass on Neuron
            y_neuron = torch.nn.functional.threshold(x_neuron, threshold, value)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::threshold")

        # Check that results match
        assert y_neuron.device.type == "neuron"
        assert y_neuron.dtype == dtype
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(y_neuron.cpu(), y, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("threshold,value", THRESHOLD_VALUE_PAIRS)
    def test_threshold_backward(self, shape, dtype, threshold, value):
        """Test threshold backward operation with various shapes, dtypes, and
        threshold/value pairs."""
        # Create input tensor that requires grad
        x = torch.randn(shape, dtype=dtype) * 2
        x.requires_grad = True
        x_neuron = x.detach().clone().to("neuron")
        x_neuron.requires_grad = True

        with track_neuron_ops():
            # Forward pass
            y = torch.nn.functional.threshold(x, threshold, value)
            y_neuron = torch.nn.functional.threshold(x_neuron, threshold, value)

            # Create gradient tensor
            grad_output = torch.randn_like(y)
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            # Verify operations ran on Neuron
            assert_op_runs_on_neuron("aten::threshold")
            assert_op_runs_on_neuron("aten::threshold_backward")

        # Check that gradients match
        assert x_neuron.grad.device.type == "neuron"
        assert x_neuron.grad.dtype == dtype
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-4
        print(f"neuron grad {x_neuron.grad.cpu()}")
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.skipif(
    not use_mlir_aten_ops(), reason="aten::threshold only supported using dynamo decompositions"
)
class TestThresholdEdgeCases:
    """Test threshold operations with edge cases."""

    def test_threshold_all_above(self):
        """Test threshold where all values are above threshold."""
        x = torch.rand(4, 4) + 1.0  # All values > 0.5
        x_neuron = x.clone().to("neuron")
        threshold, value = 0.5, 0.0

        with track_neuron_ops():
            y = torch.nn.functional.threshold(x, threshold, value)
            y_neuron = torch.nn.functional.threshold(x_neuron, threshold, value)
            assert_op_runs_on_neuron("aten::threshold")

        torch.testing.assert_close(y_neuron.cpu(), y, rtol=1e-4, atol=1e-4)

    def test_threshold_all_below(self):
        """Test threshold where all values are below threshold."""
        x = torch.rand(4, 4) - 1.0  # All values < 0.5
        x_neuron = x.clone().to("neuron")
        threshold, value = 0.5, -1.0

        with track_neuron_ops():
            y = torch.nn.functional.threshold(x, threshold, value)
            y_neuron = torch.nn.functional.threshold(x_neuron, threshold, value)
            assert_op_runs_on_neuron("aten::threshold")

        torch.testing.assert_close(y_neuron.cpu(), y, rtol=1e-4, atol=1e-4)

    def test_threshold_backward_gradient_flow(self):
        """Test that gradients flow correctly through threshold."""
        x = torch.randn(3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)
        threshold, value = 0.0, 0.0

        with track_neuron_ops():
            # Forward
            y = torch.nn.functional.threshold(x, threshold, value)
            y_neuron = torch.nn.functional.threshold(x_neuron, threshold, value)

            # Backward with ones gradient
            y.backward(torch.ones_like(y))
            y_neuron.backward(torch.ones_like(y_neuron))

            assert_op_runs_on_neuron("aten::threshold")
            assert_op_runs_on_neuron("aten::threshold_backward")

        # For threshold with value=0, gradient should be 1 where x > threshold, else 0
        expected_grad = (x > threshold).float()
        torch.testing.assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(x_neuron.grad.cpu(), expected_grad, rtol=1e-4, atol=1e-4)

    def test_threshold_inplace_equivalent(self):
        """Test that threshold produces same results as manual implementation."""
        x = torch.randn(4, 4)
        x_neuron = x.clone().to("neuron")
        threshold, value = 0.3, -0.5

        with track_neuron_ops():
            y_neuron = torch.nn.functional.threshold(x_neuron, threshold, value)
            assert_op_runs_on_neuron("aten::threshold")

        # Manual implementation
        y_manual = torch.where(x > threshold, x, torch.tensor(value))

        torch.testing.assert_close(y_neuron.cpu(), y_manual, rtol=1e-4, atol=1e-4)
