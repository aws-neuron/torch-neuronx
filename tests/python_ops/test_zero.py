import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestZero:
    def test_zero_basic(self):
        """Test basic zero_ operation"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2, 3], device=device)
            x_cpu = torch.tensor([1, 2, 3])

            x_neuron.zero_()
            x_cpu.zero_()

            assert x_neuron.device.type == "neuron"
            assert torch.all(x_neuron.cpu() == 0)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::zero_")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
    def test_zero_dtypes(self, dtype):
        """Test zero_ with different dtypes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.ones(3, 4, device=device, dtype=dtype)
            x_cpu = torch.ones(3, 4, dtype=dtype)

            x_neuron.zero_()
            x_cpu.zero_()

            assert x_neuron.dtype == dtype
            assert torch.all(x_neuron.cpu() == 0)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::zero_")

    @pytest.mark.parametrize("shape", [(2, 3), (4, 5, 6), (7,), (1, 1)])
    def test_zero_shapes(self, shape):
        """Test zero_ with different shapes"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.ones(shape, device=device)
            x_cpu = torch.ones(shape)

            x_neuron.zero_()
            x_cpu.zero_()

            assert x_neuron.shape == shape
            assert torch.all(x_neuron.cpu() == 0)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::zero_")

    def test_zero_scalar_tensor(self):
        """Test zero_ with scalar tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor(5.0, device=device)
            x_cpu = torch.tensor(5.0)

            x_neuron.zero_()
            x_cpu.zero_()

            assert x_neuron.dim() == 0
            assert x_neuron.item() == 0.0

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::zero_")

    def test_zero_empty(self):
        """Test zero_ with empty tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.empty(0, 5, device=device)
            x_cpu = torch.empty(0, 5)

            x_neuron.zero_()
            x_cpu.zero_()

            assert x_neuron.shape == (0, 5)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::zero_")

    def test_zero_with_extremes(self):
        """Test zero_ with tensors containing extreme values"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([float("inf"), float("-inf"), float("nan")], device=device)
            x_cpu = torch.tensor([float("inf"), float("-inf"), float("nan")])

            x_neuron.zero_()
            x_cpu.zero_()

            assert torch.all(x_neuron.cpu() == 0)

            torch.testing.assert_close(x_neuron.cpu(), x_cpu)
            assert_op_runs_on_neuron("aten::zero_")

    def test_zero_large_tensor(self):
        """Test zero_ with a large tensor"""
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.ones(1000, 1000, device=device)

            x_neuron.zero_()

            assert x_neuron[0, 0].item() == 0
            assert x_neuron[500, 500].item() == 0
            assert x_neuron[999, 999].item() == 0
            assert_op_runs_on_neuron("aten::zero_")

    def test_zero_tensor_slice_reference(self):
        device = "neuron"
        with track_neuron_ops():
            x_neuron = torch.tensor([1, 2], device=device)

            x_neuron_slice = x_neuron[1]
            x_neuron.zero_()
            assert x_neuron_slice.item() == 0
            assert_op_runs_on_neuron("aten::zero_")
