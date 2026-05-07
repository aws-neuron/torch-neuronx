"""Test that sum operation is properly registered with PyTorch dispatcher."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestSum:
    """Test suite for sum operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(16,), (4, 5), (4, 5, 6)])
    def test_sum_basic_shapes(self, device, shape):
        """Test sum with various tensor shapes."""
        with track_neuron_ops():
            x = torch.ones(shape, device=device)
            neuron_result = torch.sum(x)
            cpu_result = torch.sum(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum")

    @pytest.mark.parametrize(
        "shape,dim", [((4, 5, 6), 0), ((4, 5, 6), 1), ((4, 5, 6), -1), ((), 0)]
    )
    def test_sum_with_dim(self, device, shape, dim):
        """Test sum with specified dimension."""
        with track_neuron_ops():
            x = torch.ones(shape, device=device)
            neuron_result = torch.sum(x, dim=dim)
            cpu_result = torch.sum(x.cpu(), dim=dim)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum.dim_IntList")

    @pytest.mark.parametrize("keepdim", [True, False])
    def test_sum_with_keepdim(self, device, keepdim):
        """Test sum with keepdim parameter."""
        with track_neuron_ops():
            x = torch.ones(4, 5, 6, device=device)
            neuron_result = torch.sum(x, dim=1, keepdim=keepdim)
            cpu_result = torch.sum(x.cpu(), dim=1, keepdim=keepdim)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum.dim_IntList")

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float64, torch.bfloat16, torch.float16, torch.int32, torch.int64],
    )
    def test_sum_different_dtypes(self, device, dtype):
        """Test sum with different data types."""
        with track_neuron_ops():
            x = torch.ones(4, 5, dtype=dtype, device=device)
            neuron_result = torch.sum(x)
            cpu_result = torch.sum(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum")

    def test_sum_with_dtype_conversion(self, device):
        """Test sum with dtype parameter."""
        with track_neuron_ops():
            x = torch.ones(16, 16, dtype=torch.float16, device=device)
            neuron_result = torch.sum(x, dtype=torch.float32)
            cpu_result = torch.sum(x.cpu(), dtype=torch.float32)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == torch.float32
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum")

    def test_sum_with_out(self, device):
        """Test sum with output tensor."""
        with track_neuron_ops():
            x = torch.ones(4, 5, 6, device=device)
            out = torch.empty(4, 6, device=device)
            neuron_result = torch.sum(x, dim=1, out=out)

            cpu_x = x.cpu()
            cpu_out = torch.empty(4, 6)
            torch.sum(cpu_x, dim=1, out=cpu_out)

            assert neuron_result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_out)
            assert_op_runs_on_neuron("aten::sum.IntList_out")

    def test_sum_multiple_dims(self, device):
        """Test sum with multiple dimensions."""
        with track_neuron_ops():
            x = torch.ones(4, 5, 6, device=device)
            neuron_result = torch.sum(x, dim=(1, 2))
            cpu_result = torch.sum(x.cpu(), dim=(1, 2))

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum.dim_IntList")

    def test_sum_boolean_input(self, device):
        """Test sum with boolean input tensor."""
        with track_neuron_ops():
            x = torch.tensor(
                [[True, False, True], [False, True, True]], dtype=torch.bool, device=device
            )
            neuron_result = torch.sum(x)
            cpu_result = torch.sum(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum")

    def test_sum_method(self, device):
        """Test x.sum method."""
        with track_neuron_ops():
            x = torch.ones(16, 16, device=device)
            neuron_result = x.sum()
            cpu_result = x.cpu().sum()

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum")

    def test_sum_method_with_dim(self, device):
        """Test x.sum method with dimension."""
        with track_neuron_ops():
            x = torch.ones(4, 5, 6, device=device)
            neuron_result = x.sum(dim=1)
            cpu_result = x.cpu().sum(dim=1)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum.dim_IntList")

    @pytest.mark.parametrize("dtype", [None, torch.float32])
    def test_sum_scalar(self, device, dtype):
        """Test sum on scalar tensor."""
        with track_neuron_ops():
            x = torch.tensor(3.0, dtype=torch.float16, device=device)
            neuron_result = torch.sum(x, dtype=dtype)
            cpu_result = torch.sum(x.cpu(), dtype=dtype)

            assert neuron_result.device.type == "neuron"
            if dtype:
                assert neuron_result.dtype == dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::sum")

    @pytest.mark.parametrize("invalid_dim", [[1], 1])
    @assert_raises(IndexError, match="Dimension out of range")
    def test_sum_scalar_invalid_dim(self, device, invalid_dim):
        """Test sum on scalar tensor with invalid dimension raises IndexError."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            torch.sum(x, dim=invalid_dim)

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_sum_scalar_keepdim(self, device, keepdim):
        """Test sum on scalar tensor respects keepdim."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            neuron_result = torch.sum(x, dim=0, keepdim=keepdim)
            cpu_result = torch.sum(x.cpu(), dim=0, keepdim=keepdim)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    @pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.bool])
    def test_sum_integer_promotes_to_int64(self, device, dtype):
        """Verify integer/bool sum promotes output dtype to int64."""
        x = torch.ones(4, 4, dtype=dtype, device=device)
        result = torch.sum(x)
        expected = torch.sum(x.cpu())
        assert (
            result.dtype == expected.dtype
        ), f"dtype={dtype}: neuron={result.dtype}, cpu={expected.dtype}"

    def test_sum_dim_integer_promotes_to_int64(self, device):
        """Verify integer sum with dim promotes output dtype to int64."""
        x = torch.ones(4, 4, dtype=torch.int32, device=device)
        result = torch.sum(x, dim=0)
        expected = torch.sum(x.cpu(), dim=0)
        assert result.dtype == expected.dtype
