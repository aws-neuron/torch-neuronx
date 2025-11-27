"""Tests for amax and amin operations."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestAmaxAmin:
    """Test suite for amax and amin operations."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5)])
    def test_op_basic_shapes(self, device, op_name, op_func, shape):
        """Test operation with various tensor shapes."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = op_func(x)
            cpu_result = op_func(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_op_with_dim(self, device, op_name, op_func, dim):
        """Test operation with specific dimensions."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            neuron_result = op_func(x, dim=dim)
            cpu_result = op_func(x.cpu(), dim=dim)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @pytest.mark.parametrize("dims", [[0], [1], [0, 1], [-1, -2]])
    def test_op_with_multiple_dims(self, device, op_name, op_func, dims):
        """Test operation with multiple dimensions."""
        with track_neuron_ops():
            x = torch.randn(3, 4, 5, device=device)
            neuron_result = op_func(x, dim=dims)
            cpu_result = op_func(x.cpu(), dim=dims)

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_op_keepdim(self, device, op_name, op_func, keepdim):
        """Test operation with keepdim parameter."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            neuron_result = op_func(x, dim=1, keepdim=keepdim)
            cpu_result = op_func(x.cpu(), dim=1, keepdim=keepdim)

            assert neuron_result.device.type == "neuron"
            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32, torch.int64])
    def test_op_different_dtypes(self, device, op_name, op_func, dtype):
        """Test operation with different data types."""
        with track_neuron_ops():
            x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
            neuron_result = op_func(x)
            cpu_result = op_func(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == cpu_result.dtype
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    def test_op_single_element(self, device, op_name, op_func):
        """Test operation with single element tensor."""
        with track_neuron_ops():
            x = torch.tensor([5.0], device=device)
            neuron_result = op_func(x)
            cpu_result = op_func(x.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    def test_op_with_out_same_dtype(self, device, op_name, op_func):
        """Test operation with out argument of same dtype."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            out = torch.empty((), device=device)
            result = op_func(x, out=out)
            cpu_result = op_func(x.cpu())

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @assert_raises(
        TypeError,
        match="Expected the dtype for input and out to match, "
        "but got torch.int32 for input's dtype and torch.float32 for out's dtype.",
    )
    def test_op_with_out_different_dtype(self, device, op_name, op_func):
        """Test operation with out argument of different dtype."""
        x = torch.randint(0, 10, (3, 4), dtype=torch.int32, device=device)
        out = torch.empty((), dtype=torch.float32, device=device)
        op_func(x, out=out)

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @assert_raises(
        RuntimeError,
        match="Expected the dtype for input and out to match, "
        "but got Int for input's dtype and Float for out's dtype.",
    )
    def test_op_with_out_different_dtype_cpu(self, device, op_name, op_func):
        """Test operation with out argument of different dtype on cpu."""
        x = torch.randint(0, 10, (3, 4), dtype=torch.int32, device=device)
        out = torch.empty((), dtype=torch.float32, device=device)

        # Make sure CPU also fails in case PyTorch starts support dtype casting
        op_func(x.cpu(), out=out.cpu())

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    def test_op_with_out_dim_keepdim(self, device, op_name, op_func):
        """Test operation with out argument, dim and keepdim."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            out = torch.empty((3, 1), device=device)
            result = op_func(x, dim=1, keepdim=True, out=out)
            cpu_result = op_func(x.cpu(), dim=1, keepdim=True)

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron(f"aten::{op_name}")
