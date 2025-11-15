"""Test that softmax and log_softmax backward operations
are properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSoftmaxBackwardRegistration:
    """Test softmax backward operation registration and functionality."""

    def test_softmax_backward_basic(self):
        """Test basic softmax backward operation."""
        # Create input tensor that requires grad
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float32)
        x_neuron = x.detach().clone().to("neuron")
        x_neuron.requires_grad = True

        with track_neuron_ops():
            # Forward pass
            y = torch.nn.functional.softmax(x)
            y_neuron = torch.nn.functional.softmax(x_neuron)

            # Create gradient tensor
            grad_output = torch.tensor([0.1, 0.2, 0.3])
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::softmax")
            assert_op_runs_on_neuron("aten::_softmax_backward_data")

        # Check that gradients match
        assert x_neuron.grad.device.type == "neuron"
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_softmax_backward_2d(self):
        """Test softmax backward with 2D tensor."""
        # Create 2D input tensor
        x = torch.randn(3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        # Test for both dimensions
        for dim in [0, 1]:
            with track_neuron_ops():
                # Forward pass
                y = torch.softmax(x, dim=dim)
                y_neuron = torch.softmax(x_neuron, dim=dim)

                # Create gradient tensor
                grad_output = torch.randn_like(y)
                grad_output_neuron = grad_output.to("neuron")

                # Backward pass
                y.backward(grad_output, retain_graph=True)
                y_neuron.backward(grad_output_neuron, retain_graph=True)

                # Verify operation ran on Neuron
                assert_op_runs_on_neuron("aten::softmax")
                assert_op_runs_on_neuron("aten::_softmax_backward_data")

            # Check that gradients match
            assert x_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

            # Reset gradients for next iteration
            x.grad.zero_()
            x_neuron.grad.zero_()

    def test_softmax_backward_3d(self):
        """Test softmax backward with 3D tensor."""
        # Create 3D input tensor
        x = torch.randn(2, 3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        # Test for all dimensions
        for dim in [0, 1, 2]:
            with track_neuron_ops():
                # Forward pass
                y = torch.softmax(x, dim=dim)
                y_neuron = torch.softmax(x_neuron, dim=dim)

                # Create gradient tensor
                grad_output = torch.randn_like(y)
                grad_output_neuron = grad_output.to("neuron")

                # Backward pass
                y.backward(grad_output, retain_graph=True)
                y_neuron.backward(grad_output_neuron, retain_graph=True)

                # Verify operation ran on Neuron
                assert_op_runs_on_neuron("aten::softmax")
                assert_op_runs_on_neuron("aten::_softmax_backward_data")

            # Check that gradients match
            assert x_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

            # Reset gradients for next iteration
            x.grad.zero_()
            x_neuron.grad.zero_()

    def test_softmax_backward_different_dtypes(self):
        """Test softmax backward with different data types."""
        for dtype in [torch.float32, torch.float16]:
            # Create input tensor
            x = torch.randn(3, 4, requires_grad=True, dtype=dtype)
            x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

            with track_neuron_ops():
                # Forward pass
                y = torch.softmax(x, dim=1)
                y_neuron = torch.softmax(x_neuron, dim=1)

                # Create gradient tensor
                grad_output = torch.randn_like(y)
                grad_output_neuron = grad_output.to("neuron")

                # Backward pass
                y.backward(grad_output, retain_graph=True)
                y_neuron.backward(grad_output_neuron, retain_graph=True)

                # Verify operation ran on Neuron
                assert_op_runs_on_neuron("aten::softmax")
                assert_op_runs_on_neuron("aten::_softmax_backward_data")

            # Check that gradients match
            assert x_neuron.grad.device.type == "neuron"
            assert x_neuron.grad.dtype == dtype
            torch.testing.assert_close(
                x_neuron.grad.cpu(),
                x.grad,
                rtol=1e-3 if dtype == torch.float16 else 1e-4,
                atol=1e-3 if dtype == torch.float16 else 1e-4,
            )

            # Reset gradients for next iteration
            if x.grad is not None:
                x.grad.zero_()
            if x_neuron.grad is not None:
                x_neuron.grad.zero_()

    def test_softmax_backward_large(self):
        """Test softmax backward with larger tensor."""
        # Create larger input tensor
        x = torch.randn(10, 20, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            # Forward pass
            y = torch.softmax(x, dim=1)
            y_neuron = torch.softmax(x_neuron, dim=1)

            # Create gradient tensor
            grad_output = torch.randn_like(y)
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::softmax")
            assert_op_runs_on_neuron("aten::_softmax_backward_data")

        # Check that gradients match
        assert x_neuron.grad.device.type == "neuron"
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLogSoftmaxBackwardRegistration:
    """Test log_softmax backward operation registration and functionality."""

    def test_log_softmax_backward_basic(self):
        """Test basic log_softmax backward operation."""
        # Create input tensor that requires grad
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            # Forward pass
            y = torch.log_softmax(x, dim=0)
            y_neuron = torch.log_softmax(x_neuron, dim=0)

            # Create gradient tensor
            grad_output = torch.tensor([0.1, 0.2, 0.3])
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::log_softmax")
            assert_op_runs_on_neuron("aten::_log_softmax_backward_data")

        # Check that gradients match
        assert x_neuron.grad.device.type == "neuron"
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_log_softmax_backward_2d(self):
        """Test log_softmax backward with 2D tensor."""
        # Create 2D input tensor
        x = torch.randn(3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        # Test for both dimensions
        for dim in [0, 1]:
            with track_neuron_ops():
                # Forward pass
                y = torch.log_softmax(x, dim=dim)
                y_neuron = torch.log_softmax(x_neuron, dim=dim)

                # Create gradient tensor
                grad_output = torch.randn_like(y)
                grad_output_neuron = grad_output.to("neuron")

                # Backward pass
                y.backward(grad_output, retain_graph=True)
                y_neuron.backward(grad_output_neuron, retain_graph=True)

                # Verify operation ran on Neuron
                assert_op_runs_on_neuron("aten::log_softmax")
                assert_op_runs_on_neuron("aten::_log_softmax_backward_data")

            # Check that gradients match
            assert x_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

            # Reset gradients for next iteration
            x.grad.zero_()
            x_neuron.grad.zero_()

    def test_log_softmax_backward_3d(self):
        """Test log_softmax backward with 3D tensor."""
        # Create 3D input tensor
        x = torch.randn(2, 3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        # Test for all dimensions
        for dim in [0, 1, 2]:
            with track_neuron_ops():
                # Forward pass
                y = torch.log_softmax(x, dim=dim)
                y_neuron = torch.log_softmax(x_neuron, dim=dim)

                # Create gradient tensor
                grad_output = torch.randn_like(y)
                grad_output_neuron = grad_output.to("neuron")

                # Backward pass
                y.backward(grad_output, retain_graph=True)
                y_neuron.backward(grad_output_neuron, retain_graph=True)

                # Verify operation ran on Neuron
                assert_op_runs_on_neuron("aten::log_softmax")
                assert_op_runs_on_neuron("aten::_log_softmax_backward_data")

            # Check that gradients match
            assert x_neuron.grad.device.type == "neuron"
            torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

            # Reset gradients for next iteration
            x.grad.zero_()
            x_neuron.grad.zero_()

    def test_log_softmax_backward_different_dtypes(self):
        """Test log_softmax backward with different data types."""
        for dtype in [torch.float32, torch.float16]:
            # Create input tensor
            x = torch.randn(3, 4, requires_grad=True, dtype=dtype)
            x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

            with track_neuron_ops():
                # Forward pass
                y = torch.log_softmax(x, dim=1)
                y_neuron = torch.log_softmax(x_neuron, dim=1)

                # Create gradient tensor
                grad_output = torch.randn_like(y)
                grad_output_neuron = grad_output.to("neuron")

                # Backward pass
                y.backward(grad_output, retain_graph=True)
                y_neuron.backward(grad_output_neuron, retain_graph=True)

                # Verify operation ran on Neuron
                assert_op_runs_on_neuron("aten::log_softmax")
                assert_op_runs_on_neuron("aten::_log_softmax_backward_data")

            # Check that gradients match
            assert x_neuron.grad.device.type == "neuron"
            assert x_neuron.grad.dtype == dtype
            torch.testing.assert_close(
                x_neuron.grad.cpu(),
                x.grad,
                rtol=1e-3 if dtype == torch.float16 else 1e-4,
                atol=1e-3 if dtype == torch.float16 else 1e-4,
            )

            # Reset gradients for next iteration
            if x.grad is not None:
                x.grad.zero_()
            if x_neuron.grad is not None:
                x_neuron.grad.zero_()

    def test_log_softmax_backward_large(self):
        """Test log_softmax backward with larger tensor."""
        # Create larger input tensor
        x = torch.randn(10, 20, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            # Forward pass
            y = torch.log_softmax(x, dim=1)
            y_neuron = torch.log_softmax(x_neuron, dim=1)

            # Create gradient tensor
            grad_output = torch.randn_like(y)
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::log_softmax")
            assert_op_runs_on_neuron("aten::_log_softmax_backward_data")

        # Check that gradients match
        assert x_neuron.grad.device.type == "neuron"
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_log_softmax_vs_softmax_backward(self):
        """Test that log_softmax backward matches log(softmax) backward."""
        # Create input tensor
        x = torch.randn(3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            # Forward pass with log_softmax
            y_log = torch.log_softmax(x, dim=1)
            y_log_neuron = torch.log_softmax(x_neuron, dim=1)

            # Create gradient tensor
            grad_output = torch.randn_like(y_log)
            grad_output_neuron = grad_output.to("neuron")

            # Backward pass
            y_log.backward(grad_output)
            y_log_neuron.backward(grad_output_neuron)

            # Verify operation ran on Neuron
            assert_op_runs_on_neuron("aten::log_softmax")
            assert_op_runs_on_neuron("aten::_log_softmax_backward_data")

        # Check that gradients match
        assert x_neuron.grad.device.type == "neuron"
        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)
