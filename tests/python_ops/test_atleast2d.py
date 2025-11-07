import pytest
import torch

import torch_neuron
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestAtLeast2D:
    """Test cases for aten::atleast_2d implementation"""

    def test_atleast_2d_scalar(self):
        """Test atleast_2d with scalar input"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor(5, device=device)
            x_cpu = torch.tensor(5)

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::atleast_2d")

    @pytest.mark.parametrize(
        "input_shape",
        [
            (),  # scalar
            (5,),  # 1D
            (3, 4),  # 2D unchanged
            (2, 3, 4),  # 3D unchanged
            (1, 1, 1, 1),  # 4D unchanged
        ],
    )
    def test_atleast_2d_shapes(self, input_shape):
        """Parameterized test for various input shapes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(input_shape, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::atleast_2d")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_atleast_2d_dtypes(self, dtype):
        """Test atleast_2d with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3], device=device, dtype=dtype)
            x_cpu = torch.tensor([1, 2, 3], dtype=dtype)

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::atleast_2d")

    def test_atleast_2d_empty_1d(self):
        """Test atleast_2d with empty 1-D tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([], device=device)
            x_cpu = torch.tensor([])

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::atleast_2d")

    def test_atleast_2d_empty_2d(self):
        """Test atleast_2d with empty 2-D tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.empty((0, 5), device=device)
            x_cpu = torch.empty((0, 5))

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::atleast_2d")

    def test_atleast_2d_negative_values(self):
        """Test atleast_2d with negative values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([-1, -2, -3], device=device)
            x_cpu = torch.tensor([-1, -2, -3])

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_runs_on_neuron("aten::atleast_2d")

    def test_atleast_2d_inf_nan_values(self):
        """Test atleast_2d with infinity and NaN values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            x_cpu = torch.tensor([float("inf"), float("-inf"), float("nan")])

            result_neuron = torch.atleast_2d(x_neuron)
            result_cpu = torch.atleast_2d(x_cpu)

            # Verify inf values
            assert result_neuron[0, 0].cpu() == float("inf")
            assert result_neuron[0, 1].cpu() == float("-inf")

            # Verify NaN value
            assert torch.isnan(result_neuron[0, 2].cpu())
            assert torch.isnan(result_cpu[0, 2])

            assert_op_runs_on_neuron("aten::atleast_2d")
