import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_does_not_run, track_neuron_ops


class TestUnsqueeze:
    """Test cases for aten::unsqueeze implementation

    Note: unsqueeze is a metadata-only operation that should not run on neuron.
    """

    @pytest.mark.parametrize("dim", [0, 1, -1, -2])
    def test_unsqueeze_1d(self, dim):
        """Test unsqueeze with 1D tensor at different dimensions"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3], device=device)
            x_cpu = torch.tensor([1, 2, 3])

            result_neuron = torch.unsqueeze(x_neuron, dim)
            result_cpu = torch.unsqueeze(x_cpu, dim)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_does_not_run("aten::unsqueeze")

    @pytest.mark.parametrize(
        "input_shape,dim",
        [
            ((3, 4), 0),  # 2D tensor, dim 0
            ((3, 4), 1),  # 2D tensor, dim 1
            ((3, 4), 2),  # 2D tensor, dim 2
            ((3, 4), -1),  # 2D tensor, dim -1
            ((2, 3, 4), 1),  # 3D tensor, dim 1
            ((2, 3, 4), -2),  # 3D tensor, dim -2
        ],
    )
    def test_unsqueeze_multidim(self, input_shape, dim):
        """Test unsqueeze with multi-dimensional tensors"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.randn(input_shape, device=device)
            x_cpu = x_neuron.cpu()

            result_neuron = torch.unsqueeze(x_neuron, dim)
            result_cpu = torch.unsqueeze(x_cpu, dim)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_does_not_run("aten::unsqueeze")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_unsqueeze_dtypes(self, dtype):
        """Test unsqueeze with different data types"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3], device=device, dtype=dtype)
            x_cpu = torch.tensor([1, 2, 3], dtype=dtype)

            result_neuron = torch.unsqueeze(x_neuron, 0)
            result_cpu = torch.unsqueeze(x_cpu, 0)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_does_not_run("aten::unsqueeze")

    def test_unsqueeze_scalar(self):
        """Test unsqueeze with scalar tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor(5.0, device=device)
            x_cpu = torch.tensor(5.0)

            result_neuron = torch.unsqueeze(x_neuron, 0)
            result_cpu = torch.unsqueeze(x_cpu, 0)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_does_not_run("aten::unsqueeze")

    def test_unsqueeze_empty_tensor(self):
        """Test unsqueeze with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.empty(0, device=device)
            x_cpu = torch.empty(0)

            result_neuron = torch.unsqueeze(x_neuron, 0)
            result_cpu = torch.unsqueeze(x_cpu, 0)

            torch.testing.assert_close(result_neuron.cpu(), result_cpu)
            assert_op_does_not_run("aten::unsqueeze")

    def test_unsqueeze_method_vs_function(self):
        """Test that tensor.unsqueeze() method works same as torch.unsqueeze()"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3], device=device)
            x_cpu = torch.tensor([1, 2, 3])

            # Test method
            result_method_neuron = x_neuron.unsqueeze(0)
            result_method_cpu = x_cpu.unsqueeze(0)

            # Test function
            result_func_neuron = torch.unsqueeze(x_neuron, 0)
            result_func_cpu = torch.unsqueeze(x_cpu, 0)

            torch.testing.assert_close(result_method_neuron.cpu(), result_method_cpu)
            torch.testing.assert_close(result_func_neuron.cpu(), result_func_cpu)
            torch.testing.assert_close(result_method_neuron.cpu(), result_func_neuron.cpu())

            assert_op_does_not_run("aten::unsqueeze")
