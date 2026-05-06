import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestValueSelectingReductionBackward:
    """Test value selecting reduction backward operations"""

    @pytest.mark.parametrize(
        "dim,keepdim",
        [
            (0, True),
            (1, True),
            (2, True),
            (-1, True),
            (0, False),
            (1, False),
            (2, False),
            (-1, False),
        ],
    )
    def test_max_backward_dims(self, dim, keepdim):
        """Test max backward with different dimensions and keepdim."""
        x = torch.randn(3, 4, 5, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.max(x, dim=dim, keepdim=keepdim)
            y_neuron, indices_neuron = torch.max(x_neuron, dim=dim, keepdim=keepdim)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    def test_min_backward_dims(self, dim):
        """Test min backward with different dimensions"""
        x = torch.randn(2, 3, 4, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.min(x, dim=dim, keepdim=True)
            y_neuron, indices_neuron = torch.min(x_neuron, dim=dim, keepdim=True)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_median_backward_2d(self):
        """Test median backward on 2D tensor"""
        x = torch.randn(4, 6, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.median(x, dim=0, keepdim=False)
            y_neuron, indices_neuron = torch.median(x_neuron, dim=0, keepdim=False)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_kthvalue_backward_different_k(self, k):
        """Test kthvalue backward with different k values"""
        x = torch.randn(3, 5, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.kthvalue(x, k=k, dim=1, keepdim=True)
            y_neuron, indices_neuron = torch.kthvalue(x_neuron, k=k, dim=1, keepdim=True)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_mode_backward_dim0(self):
        """Test mode backward along dimension 0"""
        x = torch.randint(0, 3, (4, 3), dtype=torch.float32, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.mode(x, dim=0, keepdim=True)
            y_neuron, indices_neuron = torch.mode(x_neuron, dim=0, keepdim=True)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_nanmedian_backward_3d(self):
        """Test nanmedian backward on 3D tensor"""
        x = torch.randn(2, 3, 4)
        # Add some NaN values before requiring grad
        x[0, 1, 2] = float("nan")
        x.requires_grad_(True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.nanmedian(x, dim=2, keepdim=False)
            y_neuron, indices_neuron = torch.nanmedian(x_neuron, dim=2, keepdim=False)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_max_backward_1d(self):
        """Test max backward on 1D tensor"""
        x = torch.randn(10, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.max(x, dim=0, keepdim=True)
            y_neuron, indices_neuron = torch.max(x_neuron, dim=0, keepdim=True)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)

    def test_min_backward_4d(self):
        """Test min backward on 4D tensor"""
        x = torch.randn(2, 3, 4, 5, requires_grad=True)
        x_neuron = x.detach().clone().to("neuron").requires_grad_(True)

        with track_neuron_ops():
            y, indices = torch.min(x, dim=2, keepdim=False)
            y_neuron, indices_neuron = torch.min(x_neuron, dim=2, keepdim=False)

            grad_output = torch.randn(tuple(y.shape), dtype=y.dtype, device=y.device)
            grad_output_neuron = grad_output.to("neuron")

            y.backward(grad_output)
            y_neuron.backward(grad_output_neuron)

            assert_op_runs_on_neuron("aten::value_selecting_reduction_backward")

        torch.testing.assert_close(x_neuron.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)
