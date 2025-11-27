import os

import pytest
import torch

import torch_neuronx  # Registers neuron backend and ops
from tests.utils.neuron_test_utils import (
    assert_op_falls_back_on_cpu,
    cpu_fallback_test,
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


class TestCpuFallback:
    @cpu_fallback_test
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

    @cpu_fallback_test
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

    @cpu_fallback_test
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

    @cpu_fallback_test
    def test_static_arg_single_int(self):
        """Test CPU fallback with single int static arg: index_select(x, dim, index)."""
        x = torch.randn(3, 4, device="neuron")
        index = torch.tensor([0, 2], dtype=torch.int64, device="neuron")

        with track_neuron_ops():
            result = torch.index_select(x, 1, index)
            assert result.shape == (3, 2)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::index_select")

    @cpu_fallback_test
    def test_static_arg_tuple_string_float(self):
        """Test CPU fallback with tuple, string and float as static arg: pad(x, pad, mode, value)"""
        x = torch.randn(2, 3, device="neuron")

        with track_neuron_ops():
            result = torch.nn.functional.pad(x, (1, 1), mode="constant", value=2.0)
            assert result.shape == (2, 5)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::pad")

    @cpu_fallback_test
    def test_static_arg_bool(self):
        """Test CPU fallback with bool as static args: triu with diagonal."""
        m = torch.randn(4, 4, device="neuron")

        with track_neuron_ops():
            # diagonal=1 is int static arg, but tests bool path in conversion
            result = torch.triu(m, diagonal=1)
            assert result.shape == (4, 4)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::triu")

    @cpu_fallback_test
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

    @cpu_fallback_test
    def test_static_arg_stack(self):
        """Test CPU fallback with list of input tensors: stack(tensors, dim)."""
        t1 = torch.randn(2, 3, device="neuron")
        t2 = torch.randn(2, 3, device="neuron")

        with track_neuron_ops():
            result = torch.stack([t1, t2], dim=1)
            assert result.shape == (2, 2, 3)
            assert result.device.type == "neuron"
            assert_op_falls_back_on_cpu("aten::stack")

    @cpu_fallback_test
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

    @cpu_fallback_test
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
