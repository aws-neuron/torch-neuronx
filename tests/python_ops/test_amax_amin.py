"""Tests for amax and amin operations."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)

# The Neuron min and max reductions drop NaN, so argmin/argmax return the
# real-extremum index, not the first-NaN index CPU returns.
_ARGMAX_NAN = "argmax NaN index differs from CPU: the Neuron max reduction drops NaN"
_ARGMIN_NAN = "argmin NaN index differs from CPU: the Neuron min reduction drops NaN"


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
    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((3, 4), 0),
            ((3, 4), 1),
            ((3, 4), -1),
            ((), 0),
        ],
    )
    def test_op_with_dim(self, device, op_name, op_func, shape, dim):
        """Test operation with specific dimensions."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
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

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @assert_raises(IndexError, match="Dimension out of range")
    def test_op_scalar_invalid_dim(self, device, op_name, op_func):
        """Test operation on scalar tensor with invalid dimension."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            op_func(x, dim=[1], keepdim=True)

    @pytest.mark.parametrize("op_name,op_func", [("amax", torch.amax), ("amin", torch.amin)])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_op_scalar_keepdim(self, device, op_name, op_func, keepdim):
        """Test operation on scalar tensor respects keepdim."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            neuron_result = op_func(x, dim=0, keepdim=keepdim)
            cpu_result = op_func(x.cpu(), dim=0, keepdim=keepdim)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestArgmax:
    """Test suite for the argmax operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (1, 128), (7, 19)])
    def test_flatten(self, device, shape):
        """Test argmax over the flattened tensor (dim=None)."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.argmax(x)
            cpu_result = torch.argmax(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == torch.int64
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    @pytest.mark.parametrize("shape,dim", [((3, 4), 0), ((3, 4), 1), ((3, 4), -1), ((2, 3, 4), 1)])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_with_dim(self, device, shape, dim, keepdim):
        """Test argmax along a dimension with keepdim."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.argmax(x, dim=dim, keepdim=keepdim)
            cpu_result = torch.argmax(x.cpu(), dim=dim, keepdim=keepdim)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    )
    def test_dtypes(self, device, dtype):
        """Test argmax with the dtypes aten::argmax accepts; the index is int64."""
        with track_neuron_ops():
            if dtype.is_floating_point:
                x = torch.randn(4, 5, device=device).to(dtype)
            else:
                x = torch.randint(-9, 9, (4, 5), dtype=dtype, device=device)
            neuron_result = torch.argmax(x, dim=1)
            cpu_result = torch.argmax(x.cpu(), dim=1)

            assert neuron_result.dtype == torch.int64
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    def test_large(self, device):
        """Test argmax on a large tensor."""
        with track_neuron_ops():
            x = torch.randn(1024, 1024, device=device)
            neuron_result = torch.argmax(x, dim=1)
            cpu_result = torch.argmax(x.cpu(), dim=1)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    def test_empty_non_reduced_dim(self, device):
        """Test argmax over a non-empty dim of a tensor with a zero-size dim."""
        with track_neuron_ops():
            x = torch.randn(0, 3, device=device)
            neuron_result = torch.argmax(x, dim=1)
            cpu_result = torch.argmax(x.cpu(), dim=1)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_scalar(self, device, keepdim):
        """Test argmax on a scalar tensor; the only index is 0."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            neuron_result = torch.argmax(x, dim=0, keepdim=keepdim)
            cpu_result = torch.argmax(x.cpu(), dim=0, keepdim=keepdim)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    def test_out(self, device):
        """Test argmax with an out= buffer; it holds the result."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            out = torch.empty((3,), dtype=torch.int64, device=device)
            result = torch.argmax(x, dim=1, out=out)
            cpu_result = torch.argmax(x.cpu(), dim=1)

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    def test_out_noncontiguous(self, device):
        """Test argmax into a non-contiguous out= view."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            buf = torch.empty((3, 2), dtype=torch.int64, device=device)
            out = buf[:, 0]
            result = torch.argmax(x, dim=1, out=out)
            cpu_result = torch.argmax(x.cpu(), dim=1)

            assert result is out
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    def test_out_leaves_self_untouched(self, device):
        """Test that an out= call does not change the input tensor."""
        x = torch.randn(3, 4, device=device)
        x_copy = x.clone()
        out = torch.empty((3,), dtype=torch.int64, device=device)
        torch.argmax(x, dim=1, out=out)
        torch.testing.assert_close(x.cpu(), x_copy.cpu())

    @assert_raises(RuntimeError, match=r"argmax\(\): does not support bool input")
    def test_bool_raises(self, device):
        """Test that a bool input raises, matching aten."""
        x = torch.ones(4, 5, dtype=torch.bool, device=device)
        torch.argmax(x, dim=1)

    @pytest.mark.parametrize(
        "extremal",
        [
            0.0,
            1.0,
            pytest.param(float("nan"), marks=pytest.mark.xfail(strict=True, reason=_ARGMAX_NAN)),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_extremal_values(self, device, extremal):
        """Test argmax on an extremal value; a nan index differs from CPU (see xfail)."""
        with track_neuron_ops():
            base = torch.tensor([0.3, 0.9, 0.0, 0.2, 0.1], device=device)
            base[2] = extremal
            neuron_result = torch.argmax(base)
            cpu_result = torch.argmax(base.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    @pytest.mark.xfail(strict=True, reason=_ARGMAX_NAN)
    def test_nan_first_index_along_dim(self, device):
        """Test the first-nan index along a dim; it differs from CPU (see xfail)."""
        with track_neuron_ops():
            x = torch.tensor(
                [[1.0, float("nan"), 3.0], [float("nan"), 2.0, float("nan")]], device=device
            )
            neuron_result = torch.argmax(x, dim=1)
            cpu_result = torch.argmax(x.cpu(), dim=1)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmax")

    @assert_raises(RuntimeError, match=r"only tensors with up to 64 dims are supported")
    def test_ndim_over_64_raises(self, device):
        """Test that a tensor with more than 64 dims raises for a dim reduction.

        Build the tensor on CPU and transfer it. A device ``ones`` with 65 dims
        fails compilation asynchronously and poisons the next test that syncs.
        """
        x = torch.ones([1] * 65).to(device)
        torch.argmax(x, dim=0)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestArgmin:
    """Test suite for the argmin operation."""

    @pytest.fixture
    def device(self):
        """Get the neuron device."""
        return torch.device("neuron")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (1, 128), (7, 19)])
    def test_flatten(self, device, shape):
        """Test argmin over the flattened tensor (dim=None)."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.argmin(x)
            cpu_result = torch.argmin(x.cpu())

            assert neuron_result.device.type == "neuron"
            assert neuron_result.dtype == torch.int64
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_flatten_keepdim(self, device, shape):
        """Test argmin over the flattened tensor with keepdim; every dim stays size 1."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.argmin(x, dim=None, keepdim=True)
            cpu_result = torch.argmin(x.cpu(), dim=None, keepdim=True)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @pytest.mark.parametrize("shape,dim", [((3, 4), 0), ((3, 4), 1), ((3, 4), -1), ((2, 3, 4), 1)])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_with_dim(self, device, shape, dim, keepdim):
        """Test argmin along a dimension with keepdim."""
        with track_neuron_ops():
            x = torch.randn(shape, device=device)
            neuron_result = torch.argmin(x, dim=dim, keepdim=keepdim)
            cpu_result = torch.argmin(x.cpu(), dim=dim, keepdim=keepdim)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    )
    def test_dtypes(self, device, dtype):
        """Test argmin with the dtypes aten::argmin accepts; the index is int64."""
        with track_neuron_ops():
            if dtype.is_floating_point:
                x = torch.randn(4, 5, device=device).to(dtype)
            else:
                x = torch.randint(-9, 9, (4, 5), dtype=dtype, device=device)
            neuron_result = torch.argmin(x, dim=1)
            cpu_result = torch.argmin(x.cpu(), dim=1)

            assert neuron_result.dtype == torch.int64
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    def test_large(self, device):
        """Test argmin on a large tensor."""
        with track_neuron_ops():
            x = torch.randn(1024, 1024, device=device)
            neuron_result = torch.argmin(x, dim=1)
            cpu_result = torch.argmin(x.cpu(), dim=1)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    def test_empty_non_reduced_dim(self, device):
        """Test argmin over a non-empty dim of a tensor with a zero-size dim."""
        with track_neuron_ops():
            x = torch.randn(0, 3, device=device)
            neuron_result = torch.argmin(x, dim=1)
            cpu_result = torch.argmin(x.cpu(), dim=1)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @pytest.mark.parametrize("keepdim", [False, True])
    def test_scalar(self, device, keepdim):
        """Test argmin on a scalar tensor; the only index is 0."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            neuron_result = torch.argmin(x, dim=0, keepdim=keepdim)
            cpu_result = torch.argmin(x.cpu(), dim=0, keepdim=keepdim)

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    def test_scalar_flatten(self, device):
        """Test argmin on a scalar tensor with dim=None; the only index is 0."""
        with track_neuron_ops():
            x = torch.tensor(3.0, device=device)
            neuron_result = torch.argmin(x)
            cpu_result = torch.argmin(x.cpu())

            assert neuron_result.shape == cpu_result.shape
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)

    def test_out(self, device):
        """Test argmin with an out= buffer; it holds the result."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            out = torch.empty((3,), dtype=torch.int64, device=device)
            result = torch.argmin(x, dim=1, out=out)
            cpu_result = torch.argmin(x.cpu(), dim=1)

            assert result is out
            assert out.device.type == "neuron"
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    def test_out_noncontiguous(self, device):
        """Test argmin into a non-contiguous out= view."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            buf = torch.empty((3, 2), dtype=torch.int64, device=device)
            out = buf[:, 0]
            result = torch.argmin(x, dim=1, out=out)
            cpu_result = torch.argmin(x.cpu(), dim=1)

            assert result is out
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    def test_out_leaves_self_untouched(self, device):
        """Test that an out= call does not change the input tensor."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            x_copy = x.clone()
            out = torch.empty((3,), dtype=torch.int64, device=device)
            torch.argmin(x, dim=1, out=out)
            torch.testing.assert_close(x.cpu(), x_copy.cpu())
            assert_op_runs_on_neuron("aten::argmin")

    def test_out_flatten_keepdim(self, device):
        """Test argmin with out=, dim=None and keepdim; every dim stays size 1."""
        with track_neuron_ops():
            x = torch.randn(3, 4, device=device)
            out = torch.empty((1, 1), dtype=torch.int64, device=device)
            result = torch.argmin(x, dim=None, keepdim=True, out=out)
            cpu_result = torch.argmin(x.cpu(), dim=None, keepdim=True)

            assert result is out
            assert out.shape == cpu_result.shape
            torch.testing.assert_close(out.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @assert_raises(
        IndexError,
        match=r"argmin\(\): Expected reduction dim to be specified for input.numel\(\) == 0.",
    )
    def test_empty_full_reduction_raises(self, device):
        """Test that a full reduction over an empty tensor raises, matching aten."""
        x = torch.randn(0, 3, device=device)
        torch.argmin(x)

    @assert_raises(RuntimeError, match=r"argmin\(\): does not support bool input")
    def test_bool_raises(self, device):
        """Test that a bool input raises, matching aten."""
        x = torch.ones(4, 5, dtype=torch.bool, device=device)
        torch.argmin(x, dim=1)

    @pytest.mark.parametrize(
        "extremal",
        [
            0.0,
            1.0,
            pytest.param(float("nan"), marks=pytest.mark.xfail(strict=True, reason=_ARGMIN_NAN)),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_extremal_values(self, device, extremal):
        """Test argmin on an extremal value; a nan index differs from CPU (see xfail)."""
        with track_neuron_ops():
            base = torch.tensor([0.3, 0.9, 0.5, 0.2, 0.1], device=device)
            base[2] = extremal
            neuron_result = torch.argmin(base)
            cpu_result = torch.argmin(base.cpu())

            assert neuron_result.device.type == "neuron"
            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @pytest.mark.xfail(strict=True, reason=_ARGMIN_NAN)
    def test_nan_first_index_along_dim(self, device):
        """Test the first-nan index along a dim; it differs from CPU (see xfail)."""
        with track_neuron_ops():
            x = torch.tensor(
                [[1.0, float("nan"), 3.0], [float("nan"), 2.0, float("nan")]], device=device
            )
            neuron_result = torch.argmin(x, dim=1)
            cpu_result = torch.argmin(x.cpu(), dim=1)

            torch.testing.assert_close(neuron_result.cpu(), cpu_result)
            assert_op_runs_on_neuron("aten::argmin")

    @assert_raises(RuntimeError, match=r"only tensors with up to 64 dims are supported")
    def test_ndim_over_64_raises(self, device):
        """Test that a tensor with more than 64 dims raises for a dim reduction.

        Build the tensor on CPU and transfer it. A device ``ones`` with 65 dims
        fails compilation asynchronously and poisons the next test that syncs.
        """
        x = torch.ones([1] * 65).to(device)
        torch.argmin(x, dim=0)
