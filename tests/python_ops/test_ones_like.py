import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestOnesLike:
    def test_ones_like_basic(self):
        """Test basic ones_like operation"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(3, 4, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.ones_like(x_neuron)
            result_cpu = torch.ones_like(x_cpu)

            assert result_neuron.device.type == "neuron"

            assert result_neuron.shape == x_neuron.shape

            assert torch.all(result_neuron.cpu() == 1)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::ones_like")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.int32, torch.int64, torch.float64]
    )
    def test_ones_like_dtypes(self, dtype):
        """Test ones_like with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randint(0, 2, (2, 3), device=device, dtype=dtype)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.ones_like(x_neuron)
            result_cpu = torch.ones_like(x_cpu)

            assert result_neuron.dtype == dtype

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::ones_like")

    def test_ones_like_direct_dtype_arg(self):
        """Test ones_like with dtype passed directly as an argument"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(3, 4, dtype=torch.float32, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.ones_like(x_neuron, dtype=torch.int32)
            result_cpu = torch.ones_like(x_cpu, dtype=torch.int32)

            assert result_neuron.dtype == torch.int32

            assert torch.all(result_cpu == 1)
            assert torch.all(result_neuron.cpu() == 1)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::ones_like")

    def test_ones_like_requires_grad(self):
        """Test ones_like with requires_grad parameter"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(3, 4, device=device)
            x_cpu = x_neuron.cpu()

            # Default should not require gradients
            result_neuron = torch.ones_like(x_neuron)
            result_cpu = torch.ones_like(x_cpu)
            assert not result_neuron.requires_grad

            # Explicitly set requires_grad=True
            result_neuron_grad = torch.ones_like(x_neuron, requires_grad=True)
            result_cpu_grad = torch.ones_like(x_cpu, requires_grad=True)
            assert result_neuron_grad.requires_grad

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            torch.testing.assert_close(result_neuron_grad.cpu(), result_cpu_grad)
            assert_op_runs_on_neuron("aten::ones_like")

    @pytest.mark.parametrize("shape", [(2, 3), (4, 5, 6), (7,), (1, 1)])
    def test_ones_like_shapes(self, shape):
        """Test ones_like with different shapes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(shape, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.ones_like(x_neuron)
            result_cpu = torch.ones_like(x_cpu)

            assert result_neuron.shape == shape

            assert torch.all(result_neuron.cpu() == 1)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::ones_like")

    def test_ones_like_empty(self):
        """Test ones_like with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.empty(0, 5, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.ones_like(x_neuron)
            result_cpu = torch.ones_like(x_cpu)

            assert result_neuron.shape == (0, 5)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::ones_like")

    def test_ones_like_scalar_tensor(self):
        """Test ones_like with scalar tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor(5.0, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.ones_like(x_neuron)
            result_cpu = torch.ones_like(x_cpu)

            assert result_neuron.dim() == 0

            assert result_neuron.item() == 1.0

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::ones_like")
