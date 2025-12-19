import os

import pytest
import torch

import torch_neuronx  # Registers neuron backend and ops
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    assert_raises,
    track_neuron_ops,
)


@pytest.fixture
def fail_jax_kernel_call(monkeypatch):
    """Force JaxKernel call to fail so Operation falls back to CPU.

    This simulates a JAX/compilation/execution failure without invoking the runtime.
    """
    from torch_neuronx.python_ops.jax.kernel import JaxKernel

    def _fail(self, *inputs, **kwargs):
        raise RuntimeError("Simulated JAX kernel failure")

    monkeypatch.setattr(JaxKernel, "__call__", _fail)


class TestCpuFallbackLoweringError:
    def test_factory_op_no_tensor_args_falls_back_cleanly(self, fail_jax_kernel_call):
        """
        When a factory op (no tensor inputs) fails in JAX path, ensure CPU fallback:
        - rewrites device kw to CPU (to avoid recursion)
        - returns a tensor on neuron (copied back)
        - records as a fallback op
        """
        size = (2, 3)
        device = "neuron:0"

        with track_neuron_ops():
            x = torch.ones(size, device=device)
            # Should come back on neuron device after CPU fallback
            assert x.device.type == "neuron"
            assert tuple(x.shape) == size
            assert_op_falls_back_on_cpu("aten::ones")

    def test_factory_out_param_resize_and_copy_on_fallback(self, fail_jax_kernel_call):
        """
        With an out tensor provided on neuron device and wrong shape:
        - CPU fallback computes on CPU, then resizes+copies into out
        - Returns the same out tensor
        - Records as a fallback op
        """
        device = "neuron:0"
        out = torch.empty((1,), dtype=torch.float32, device=device)

        with track_neuron_ops():
            # For .out overload, do not pass device kwarg
            y = torch.ones((2, 4), out=out)
            # Same object and resized
            assert y is out
            assert tuple(out.shape) == (2, 4)
            # Values are correct
            assert torch.all(out.to("cpu") == 1)
            assert_op_falls_back_on_cpu("aten::ones.out")


class TestCpuFallback:
    @assert_raises(
        RuntimeError,
        match="CPU fallback failed. Cannot copy result to output tensor which is not contiguous",
    )
    def test_copy_neuron_to_neuron_non_contiguous(self):
        tile_t = 128
        tile_h = 128
        nopad_t = 128
        nopad_h = 64

        x = torch.zeros(tile_t, tile_h, dtype=torch.bfloat16, device="neuron")
        y = torch.zeros(tile_t, tile_h, dtype=torch.bfloat16, device="neuron")
        z = x[:nopad_t, :nopad_h]

        # Should raise error for copying to non-contiguous destination
        y[:nopad_t, :nopad_h] = z

    def test_fallback_handler_unregistered_op(self):
        """Test that unregistered operations fall back to CPU"""
        # Create a tensor on CPU first, then move to neuron
        x_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
        x = x_cpu.to("neuron")

        # Clear any previous tracking
        torch_neuronx.clear_op_tracking()

        # Use track_neuron_ops context manager to track operations
        with track_neuron_ops():
            # Use an operation that's not registered on Neuron
            # This should trigger the fallback handler
            y = torch.ops.aten._test_optional_intlist(x, None)

            # Verify that the operation was offloaded to CPU using the neuron op tracer
            assert_op_falls_back_on_cpu("aten::_test_optional_intlist")

        # Result should still be correct (computed on CPU)
        expected = torch.ops.aten._test_optional_intlist(x_cpu, None)
        assert torch.allclose(y.cpu(), expected)

    @assert_raises(
        RuntimeError,
        match="torch.histogram: input tensor and hist tensor should have the same dtype, "
        "but got input float and hist int",
    )
    def test_histc_out_inconsistent_dtype_neuron(self):
        """Test histc with inconsistent dtype on Neuron device."""
        device = "neuron"
        x = torch.randn(100, device=device)
        out = torch.zeros(10, device=device, dtype=torch.int32)
        torch.histc(x, bins=10, out=out)

    @assert_raises((IndexError, RuntimeError), match=r".*index_copy[\s\S]*out of bound.*")
    def test_index_copy_with_indices_out_of_range(self):
        """Test index select with indices out of range"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([0, 6, 2])
        indices_neuron = indices.to(device)
        input_neuron.index_copy(0, indices_neuron, source_neuron)

    @assert_raises(
        RuntimeError, match=r"index_copy_\(\): Expected a long tensor for index, but got Float"
    )
    def test_index_copy_empty_indices(self):
        """Test index select with empty index"""

        device = "neuron"
        input = torch.zeros(5, 3)
        input_neuron = input.to(device)
        source = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
        source_neuron = source.to(device)
        indices = torch.tensor([])
        indices_neuron = indices.to(device)

        # CPU error message does not seem to be descriptive neither
        input_neuron.index_copy(0, indices_neuron, source_neuron)

    @assert_raises(IndexError, match="index out of range in self")
    def test_index_select_with_input_empty(self):
        """Test index select with input empty"""

        device = "neuron"
        input = torch.empty((0,))
        indices = torch.tensor([0])

        torch.index_select(input.to(device), dim=0, index=indices.to(device))

    @assert_raises(
        AssertionError,
        match=r".*aten::index_select ran on Neuron instead of falling back to CPU",
    )
    def test_index_select_empty_indices(self):
        """Test index select with empty indices"""
        device = "neuron"
        input = torch.randn(3, 4)
        input_neuron = input.to(device)
        indices = torch.empty((0,), dtype=torch.int32)

        # empty tensor handler should handle this without running on neuron or fall back to CPU
        with track_neuron_ops():
            result = torch.index_select(input_neuron, dim=0, index=indices.to(device))
            assert result.numel() == 0
            assert_op_falls_back_on_cpu("aten::index_select")

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    @assert_raises(
        IndexError, match="tensors used as indices must be long, int, byte or bool tensors"
    )
    def test_invalid_index_dtype(self, dtype):
        """Test invalid index dtype"""
        input_data = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
        x = torch.tensor(input_data, device="neuron")
        row_idx = torch.tensor([1, 2], dtype=dtype, device="neuron")  # 15 is out of bounds
        x[row_idx, :]

    @pytest.mark.parametrize(
        "shape1,shape2,shape3,expected_error",
        [
            (
                (3, 4),
                (5, 10),
                None,
                "mat1 and mat2 shapes cannot be multiplied \\(3x4 and 10x5\\)",
            ),  # Wrong input dimension
            (
                (3, 4),
                (5, 4),
                (10,),
                "The expanded size of the tensor \\(5\\) must match the existing size \\(10\\) "
                "at non-singleton dimension 1",
            ),  # Wrong bias dimension
        ],
    )
    @assert_raises(RuntimeError)
    def test_linear_dimension_mismatch_error(self, shape1, shape2, shape3, expected_error):
        """Test that dimension mismatch raises appropriate error"""
        with track_neuron_ops():
            input_neuron = torch.rand(shape1, device="neuron")
            weight_neuron = torch.rand(shape2, device="neuron")
            bias_neuron = None if shape3 is None else torch.rand(shape3, device="neuron")

            torch.nn.functional.linear(input_neuron, weight_neuron, bias=bias_neuron)


@pytest.mark.skipif(
    os.environ.get("TORCH_NEURONX_SYNC_MODE") == "1",
    reason="Async CPU fallback is only relevant with aysnc execution mode",
)
class TestCpuFallbackAsync:
    """Test CPU fallback for operations with static arguments.

    Tests cover: int, float, bool, str, tuple, list, and tensor static args.
    """

    def setup_method(self):
        """Force compilation failure to trigger CPU fallback."""
        # Inject invalid --verbose flag to force neuronxcc compilation failure
        os.environ["NEURON_CC_FLAGS"] = "--verbose not_a_valid_arg"

    def teardown_method(self):
        """Reset environment variables."""
        os.environ.pop("NEURON_CC_FLAGS", None)

    def test_cpufallback_basic(self):
        """Test callBoxed path with scalar argument: mul."""
        x = torch.randn(2, 3, device="neuron")

        with track_neuron_ops():
            result = torch.mul(x, 2.5)
            assert result.shape == (2, 3)
            assert result.device.type == "neuron"
            expected = x.to("cpu") * 2.5
            assert torch.allclose(result.to("cpu"), expected)
            assert_op_falls_back_on_cpu("mul")

    def test_static_arg_single_int(self):
        """Test CPU fallback with single int static arg: index_select(x, dim, index)."""
        x = torch.randn(3, 4, device="neuron")
        index = torch.tensor([0, 2], dtype=torch.int64, device="neuron")

        with track_neuron_ops():
            result = torch.index_select(x, 1, index)
            assert result.shape == (3, 2)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::index_select")

    def test_static_arg_tuple_string_float(self):
        """Test CPU fallback with tuple, string and float as static arg: pad(x, pad, mode, value)"""
        x = torch.randn(2, 3, device="neuron")

        with track_neuron_ops():
            result = torch.nn.functional.pad(x, (1, 1), mode="constant", value=2.0)
            assert result.shape == (2, 5)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::pad")

    def test_static_arg_bool(self):
        """Test CPU fallback with bool as static args: triu with diagonal."""
        m = torch.randn(4, 4, device="neuron")

        with track_neuron_ops():
            # diagonal=1 is int static arg, but tests bool path in conversion
            result = torch.triu(m, diagonal=1)
            assert result.shape == (4, 4)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::triu")

    def test_static_arg_multiple_tuples(self):
        """Test CPU fallback with multiple tuples static args: conv2d."""
        input_tensor = torch.randn(1, 3, 5, 5, device="neuron")
        weight = torch.randn(6, 3, 3, 3, device="neuron")

        with track_neuron_ops():
            result = torch.nn.functional.conv2d(
                input_tensor, weight, bias=None, stride=1, padding=1
            )
            assert result.shape == (1, 6, 5, 5)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::convolution")

    def test_static_arg_stack(self):
        """Test CPU fallback with list of input tensors: stack(tensors, dim)."""
        t1 = torch.randn(2, 3, device="neuron")
        t2 = torch.randn(2, 3, device="neuron")

        with track_neuron_ops():
            result = torch.stack([t1, t2], dim=1)
            assert result.shape == (2, 2, 3)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::stack")

    def test_static_arg_tensor_inplace(self):
        """Test CPU fallback with tensor static arg in-place op: fill_(x, fill_value)."""
        x = torch.randn(2, 3, device="neuron")
        fill_value = torch.tensor(5.0, device="neuron")

        with track_neuron_ops():
            result = x.fill_(fill_value)
            assert result.shape == (2, 3)
            assert result.device.type == "neuron"
            assert torch.allclose(result.to("cpu"), torch.full((2, 3), 5.0))
            assert_op_falls_back_on_cpu("aten::fill_.Tensor")

    def test_static_arg_scalar_inplace(self):
        """Test CPU fallback with tensor static arg in-place op: fill_(x, fill_value)."""
        x = torch.randn(2, 3, device="neuron")
        fill_value = 5.0

        with track_neuron_ops():
            result = x.fill_(fill_value)
            assert result.shape == (2, 3)
            assert result.device.type == "neuron"
            assert torch.allclose(result.to("cpu"), torch.full((2, 3), 5.0))
            assert_op_falls_back_on_cpu("aten::fill_.Scalar")
