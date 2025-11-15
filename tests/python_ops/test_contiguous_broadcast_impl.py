"""Tests ensuring broadcast-based contiguous path is used when applicable."""

from unittest.mock import patch

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import get_cache_size


def test_contiguous_broadcast_uses_jax_kernel(monkeypatch):
    """Broadcast case should use the JAX broadcast kernel, not the generic copy."""
    # Patch the generic contiguous symbol used by the fallback impl
    from torch_neuronx.python_ops import contiguous as contiguous_mod
    from torch_neuronx.python_ops.contiguous_broadcast import ContiguousBroadcastImpl

    jax_called = {"count": 0}
    generic_called = {"count": 0}

    # Wrap _get_kernel to intercept JAX kernel calls
    orig_get_kernel = ContiguousBroadcastImpl._get_kernel

    def wrapped_get_kernel(self):
        k = orig_get_kernel(self)

        def wrapper(*args, **kwargs):
            jax_called["count"] += 1
            return k(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(ContiguousBroadcastImpl, "_get_kernel", wrapped_get_kernel, raising=True)

    # Wrap the generic fallback to ensure it's not hit in the broadcast case
    orig_generic = contiguous_mod.contiguous_generic_kernel

    def wrapped_generic(*args, **kwargs):
        generic_called["count"] += 1
        return orig_generic(*args, **kwargs)

    monkeypatch.setattr(contiguous_mod, "contiguous_generic_kernel", wrapped_generic, raising=True)

    # Execute: broadcast should be handled by the broadcast-based path
    scalar = torch.tensor(42.0, device="neuron")
    broadcasted_view = scalar.expand(128, 256)  # All strides = 0
    assert not broadcasted_view.is_contiguous()
    assert broadcasted_view.stride() == (0, 0)

    y = broadcasted_view.contiguous()

    assert y.is_contiguous()
    assert y.shape == (128, 256)
    assert torch.allclose(y, broadcasted_view)
    assert torch.all(y == 42.0)

    # Validate paths taken
    assert jax_called["count"] >= 1, "Expected JAX broadcast kernel to be invoked"
    assert generic_called["count"] == 0, "Generic fallback should not be used for broadcast"


def test_contiguous_broadcast_autograd():
    """Broadcast path should preserve gradients."""

    scalar_grad = torch.tensor(2.71, device="neuron", requires_grad=True)
    broadcasted_grad = scalar_grad.expand(128, 256)
    yg = broadcasted_grad.contiguous()
    yg.retain_grad()
    yg.sum().backward()

    scalar_grad_cpu = torch.tensor(2.71, device="cpu", requires_grad=True)
    broadcasted_grad_cpu = scalar_grad_cpu.expand(128, 256)
    yg_cpu = broadcasted_grad_cpu.contiguous()
    yg_cpu.retain_grad()
    yg_cpu.sum().backward()

    assert torch.allclose(
        scalar_grad.grad.cpu(), scalar_grad_cpu.grad
    ), "Neuron and CPU gradients should match"

    assert torch.allclose(
        yg.grad.cpu(), yg_cpu.grad
    ), "Neuron and CPU broadcasted gradients should match"


@pytest.mark.parametrize(
    "shape",
    [
        (2,),
        (2, 3),
        (2, 3, 4),
    ],
)
def test_contiguous_broadcast_various_shapes(shape):
    """Various broadcast shapes preserve values; gradients flow correctly."""
    scalar_val = 42.0
    scalar = torch.tensor(scalar_val, device="neuron")
    broadcasted_view = scalar.expand(*shape)
    assert not broadcasted_view.is_contiguous()
    assert all(s == 0 for s in broadcasted_view.stride())

    y = broadcasted_view.contiguous()

    assert y.is_contiguous()
    assert y.shape == broadcasted_view.shape
    assert torch.allclose(y, broadcasted_view)
    assert torch.all(y == scalar_val)


def test_incorrect_fallback_to_generic_kernel(monkeypatch):
    """Test incorrect fallback to generic kernel for different target shapes"""

    from torch_neuronx.python_ops.contiguous_broadcast import ContiguousBroadcastImpl

    compile_called = {"count": 0}
    orig_get_kernel = ContiguousBroadcastImpl._get_kernel

    def wrapped_get_kernel(self):
        kernel = orig_get_kernel(self)

        def wrapper(*args, **kwargs):
            orig_cache_size = get_cache_size(kernel)
            result = kernel(*args, **kwargs)

            updated_cache_size = get_cache_size(kernel)
            if updated_cache_size > orig_cache_size:
                compile_called["count"] += 1

            return result

        return wrapper

    monkeypatch.setattr(ContiguousBroadcastImpl, "_get_kernel", wrapped_get_kernel, raising=True)

    def mock_contiguous_generic_kernel(
        src, dst, shape, src_strides, dst_strides, src_storage_offset=0
    ):
        """Mock that returns corrupted output to detect generic kernel was used"""
        dst.cpu().zero_()  # Fill with zeros instead of correct values
        return dst

    with patch(
        "torch_neuronx.python_ops.contiguous.contiguous_generic_kernel",
        side_effect=mock_contiguous_generic_kernel,
    ):
        # Test case 1
        scalar_value = 123.0
        scalar_tensor = torch.tensor(scalar_value, dtype=torch.bfloat16, device="neuron")
        broadcasted_1 = scalar_tensor.expand(4096, 2048)
        assert broadcasted_1.stride() == (0, 0)
        assert not broadcasted_1.is_contiguous()

        compile_called["count"] = 0
        result_1 = broadcasted_1.contiguous()
        assert (
            compile_called["count"] == 1
        ), f"Expected 1 compilation, got {compile_called['count']}"

        # Should not be zeros (proving broadcast kernel was used, not mock)
        assert torch.all(result_1 == scalar_value)
        assert result_1.stride() == (2048, 1)
        assert result_1.is_contiguous()

        # Step 2: This should compile a new neff and successfully execute
        broadcasted_2 = scalar_tensor.expand(2048, 4096)
        assert not broadcasted_2.is_contiguous()

        compile_called["count"] = 0
        result_2 = broadcasted_2.contiguous()
        assert (
            compile_called["count"] == 1
        ), f"Expected 1 compilation, got {compile_called['count']}"

        assert torch.all(result_2 == scalar_value)
        assert result_2.stride() == (4096, 1)
        assert result_2.is_contiguous()


def test_mixed_broadcast_falls_back_to_generic():
    """Test that mixed broadcast/non-broadcast patterns fall back to generic kernel"""

    from torch_neuronx.python_ops.contiguous import contiguous_generic_kernel

    original_generic_kernel = contiguous_generic_kernel
    generic_called = {"count": 0}

    def mock_counting_generic_kernel(
        src, dst, shape, src_strides, dst_strides, src_storage_offset=0
    ):
        generic_called["count"] += 1
        # Call the original function we stored before patching
        return original_generic_kernel(
            src, dst, shape, src_strides, dst_strides, src_storage_offset
        )

    with patch(
        "torch_neuronx.python_ops.contiguous.contiguous_generic_kernel",
        side_effect=mock_counting_generic_kernel,
    ):
        # Create a tensor with mixed stride pattern (some 0, some non-zero)
        # This should NOT be handled by broadcast implementation
        x = torch.randn(2, 3, 4, device="neuron")
        # Create a view that has some broadcasted and some non-broadcasted dimensions
        y = x[:, 0:1, :].expand(-1, 3, -1)  # Middle dim broadcasted, others not

        # This should fall back to generic kernel since it's mixed broadcast/non-broadcast
        if not y.is_contiguous():
            y.contiguous()
            assert generic_called["count"] >= 1, "Should fall back to generic for mixed patterns"
