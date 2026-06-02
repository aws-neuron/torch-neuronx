"""Tests for aten::std and aten::std_mean on Neuron."""

import pytest
import torch

from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestStd:
    @pytest.mark.parametrize("shape", [(256, 4096), (64, 512), (4, 8, 16)])
    @pytest.mark.parametrize("correction", [0, 1])
    @pytest.mark.parametrize("dim", [None, -1])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_std(self, shape, correction, dim, keepdim):
        """Test std with various shapes, corrections, dims, and keepdim."""
        if not shape and dim is not None:
            pytest.skip("0-d tensor with explicit dim is a special case tested separately")
        x_cpu = torch.randn(shape) if shape else torch.tensor(3.14)
        x_neuron = x_cpu.to("neuron")
        kwargs = {"correction": correction}
        if dim is not None:
            kwargs["dim"] = dim
            kwargs["keepdim"] = keepdim
        with track_neuron_ops():
            result = torch.std(x_neuron, **kwargs)
            assert_op_runs_on_neuron("aten::std.correction")
        expected = torch.std(x_cpu, **kwargs)
        torch.testing.assert_close(result.cpu(), expected)

    def test_std_0d_tensor_with_dim(self):
        """Test std on 0-d tensor with explicit dim."""
        x_cpu = torch.tensor(3.14)
        x_neuron = x_cpu.to("neuron")
        with track_neuron_ops():
            result = torch.std(x_neuron, dim=0, correction=0)
            assert_op_runs_on_neuron("aten::std.correction")
        expected = torch.std(x_cpu, dim=0, correction=0)
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_std_dtypes(self, dtype):
        """Test std with different dtypes."""
        x_cpu = torch.randn(64, 256, dtype=dtype)
        x_neuron = x_cpu.to("neuron")
        with track_neuron_ops():
            result = torch.std(x_neuron, dim=1)
            assert_op_runs_on_neuron("aten::std.correction")
        expected = torch.std(x_cpu, dim=1)
        assert result.dtype == dtype
        torch.testing.assert_close(result.cpu(), expected)


class TestStdMean:
    @pytest.mark.parametrize("shape", [(256, 4096), (64, 512), (4, 8, 16)])
    @pytest.mark.parametrize("correction", [0, 1])
    @pytest.mark.parametrize("dim", [None, -1])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_std_mean(self, shape, correction, dim, keepdim):
        """Test std_mean with various shapes, corrections, dims, and keepdim."""
        if not shape and dim is not None:
            pytest.skip("0-d tensor with explicit dim is a special case tested separately")
        x_cpu = torch.randn(shape) if shape else torch.tensor(3.14)
        x_neuron = x_cpu.to("neuron")
        kwargs = {"correction": correction}
        if dim is not None:
            kwargs["dim"] = dim
            kwargs["keepdim"] = keepdim
        with track_neuron_ops():
            std_n, mean_n = torch.std_mean(x_neuron, **kwargs)
            assert_op_runs_on_neuron("aten::std_mean.correction")
        std_c, mean_c = torch.std_mean(x_cpu, **kwargs)
        torch.testing.assert_close(std_n.cpu(), std_c)
        torch.testing.assert_close(mean_n.cpu(), mean_c)

    def test_std_mean_0d_tensor_with_dim(self):
        """Test std_mean on 0-d tensor with explicit dim."""
        x_cpu = torch.tensor(3.14)
        x_neuron = x_cpu.to("neuron")
        with track_neuron_ops():
            std_n, mean_n = torch.std_mean(x_neuron, dim=0, correction=0)
            assert_op_runs_on_neuron("aten::std_mean.correction")
        std_c, mean_c = torch.std_mean(x_cpu, dim=0, correction=0)
        torch.testing.assert_close(std_n.cpu(), std_c)
        torch.testing.assert_close(mean_n.cpu(), mean_c)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_std_mean_dtypes(self, dtype):
        """Test std_mean with different dtypes."""
        x_cpu = torch.randn(64, 256, dtype=dtype)
        x_neuron = x_cpu.to("neuron")
        with track_neuron_ops():
            std_n, mean_n = torch.std_mean(x_neuron, dim=1)
            assert_op_runs_on_neuron("aten::std_mean.correction")
        std_c, mean_c = torch.std_mean(x_cpu, dim=1)
        assert std_n.dtype == dtype
        torch.testing.assert_close(std_n.cpu(), std_c)
        torch.testing.assert_close(mean_n.cpu(), mean_c)
