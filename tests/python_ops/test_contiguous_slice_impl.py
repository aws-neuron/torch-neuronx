"""Tests ensuring slice-based contiguous path is used when applicable."""

import os
from unittest.mock import patch

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import get_cache_size


@pytest.mark.usefixtures("enable_metrics_for_class")
class TestContiguousSlice:
    """Test class for contiguous slice operations"""

    def test_contiguous_slice_uses_mlir_kernel(self, monkeypatch):
        """Slice case should use the slice kernel (MLIR), not the generic copy."""
        from torch_neuronx.python_ops import contiguous as contiguous_mod
        from torch_neuronx.python_ops.contiguous_slice import ContiguousSliceMLIRImpl

        slice_called = {"count": 0}
        generic_called = {"count": 0}

        impl_class = ContiguousSliceMLIRImpl
        orig_get_kernel = impl_class._get_kernel

        def wrapped_get_kernel(self, orig_get_kernel=orig_get_kernel):
            k = orig_get_kernel(self)

            def wrapper(*args, **kwargs):
                slice_called["count"] += 1
                return k(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(impl_class, "_get_kernel", wrapped_get_kernel, raising=True)

        # Wrap the generic fallback to ensure it's not hit in the slice case
        orig_generic = contiguous_mod.contiguous_generic_kernel

        def wrapped_generic(*args, **kwargs):
            generic_called["count"] += 1
            return orig_generic(*args, **kwargs)

        monkeypatch.setattr(
            contiguous_mod, "contiguous_generic_kernel", wrapped_generic, raising=True
        )

        # Execute: slice should be handled by the slice-based path
        x = torch.randn(2, 4, 8, 64, device="neuron", dtype=torch.float32)
        sliced_view = x[..., :32]  # slice from start
        assert not sliced_view.is_contiguous()

        y = sliced_view.contiguous()

        assert y.is_contiguous()
        assert y.shape == (2, 4, 8, 32)
        assert torch.allclose(y.cpu(), sliced_view.cpu())

        # Validate paths taken
        assert slice_called["count"] >= 1, "Expected MLIR slice kernel to be invoked"
        assert generic_called["count"] == 0, "Generic fallback should not be used for slice"

    def test_contiguous_slice_autograd(self):
        """slice path should preserve gradients."""
        x_grad = torch.randn(2, 4, 8, 64, device="neuron", requires_grad=True)
        sliced_grad = x_grad[..., :32]
        yg = sliced_grad.contiguous()
        yg.retain_grad()
        yg.sum().backward()

        x_grad_cpu = x_grad.detach().cpu().requires_grad_(True)
        sliced_grad_cpu = x_grad_cpu[..., :32]
        yg_cpu = sliced_grad_cpu.contiguous()
        yg_cpu.retain_grad()
        yg_cpu.sum().backward()

        assert torch.allclose(
            x_grad.grad.cpu(), x_grad_cpu.grad, atol=1e-5
        ), "Neuron and CPU gradients should match"

    @pytest.mark.parametrize(
        "slice_spec,expected_shape",
        [
            # Start slice
            ((..., slice(None, 32)), (2, 4, 8, 32)),
            # Middle slice
            ((..., slice(16, 48)), (2, 4, 8, 32)),
            # End slice
            ((..., slice(32, None)), (2, 4, 8, 32)),
        ],
    )
    def test_contiguous_slice_various_positions(self, slice_spec, expected_shape):
        """Various slice positions preserve values."""
        x = torch.randn(2, 4, 8, 64, device="cpu", dtype=torch.float32)
        x_neuron = x.to("neuron")
        sliced_view = x_neuron[slice_spec]
        assert not sliced_view.is_contiguous()

        y_neuron = sliced_view.contiguous()

        assert y_neuron.is_contiguous()
        assert y_neuron.shape == expected_shape

        y_cpu = x[slice_spec].contiguous()
        assert torch.allclose(
            y_neuron.cpu(), y_cpu
        ), f"slice spec: {slice_spec} with expected_shape: {expected_shape} failed"

    @pytest.mark.parametrize(
        "shape,axis,slice_spec",
        [
            # Different dimensions
            ((16, 32), 0, slice(4, 12)),
            ((8, 16, 32), 1, slice(4, 12)),
            ((4, 8, 16, 32), 2, slice(4, 12)),
            ((4, 8, 16, 32), 3, slice(8, 24)),
        ],
    )
    def test_contiguous_slice_different_dimensions(self, shape, axis, slice_spec):
        """slice works on different dimensions."""
        x = torch.randn(*shape, device="neuron", dtype=torch.float32)
        slices = [slice(None)] * len(shape)
        slices[axis] = slice_spec
        sliced_view = x[tuple(slices)]

        if not sliced_view.is_contiguous():
            y = sliced_view.contiguous()
            assert y.is_contiguous()
            assert torch.allclose(y.cpu(), sliced_view.cpu())

    def test_multiple_dimensions_sliced_fallback(self):
        """Test that multiple dimension slice falls back to generic kernel."""
        from torch_neuronx.python_ops.contiguous import contiguous_generic_kernel

        original_generic_kernel = contiguous_generic_kernel
        generic_called = {"count": 0}

        def mock_counting_generic_kernel(
            src, dst, shape, src_strides, dst_strides, src_storage_offset=0
        ):
            generic_called["count"] += 1
            return original_generic_kernel(
                src, dst, shape, src_strides, dst_strides, src_storage_offset
            )

        with patch(
            "torch_neuronx.python_ops.contiguous.contiguous_generic_kernel",
            side_effect=mock_counting_generic_kernel,
        ):
            # Create a tensor with multiple dimensions sliced
            x = torch.randn(8, 16, 32, device="neuron")
            # slice multiple dimensions - should NOT be handled by slice implementation
            y = x[2:6, 4:12]  # Two dimensions sliced

            if not y.is_contiguous():
                y.contiguous()
                assert (
                    generic_called["count"] >= 1
                ), "Should fall back to generic for multiple slices"

    def test_contiguous_slice_compilation_caching(self, monkeypatch):
        """Test that different slice patterns compile separate kernels."""
        from torch_neuronx.python_ops.contiguous_slice import ContiguousSliceMLIRImpl

        compile_called = {"count": 0}

        impl_class = ContiguousSliceMLIRImpl
        orig_get_kernel = impl_class._get_kernel

        def wrapped_get_kernel(self, orig_get_kernel=orig_get_kernel):
            kernel = orig_get_kernel(self)

            def wrapper(*args, **kwargs):
                orig_cache_size = get_cache_size(kernel)
                result = kernel(*args, **kwargs)
                updated_cache_size = get_cache_size(kernel)
                if updated_cache_size > orig_cache_size:
                    compile_called["count"] += 1
                return result

            return wrapper

        monkeypatch.setattr(impl_class, "_get_kernel", wrapped_get_kernel, raising=True)

        # Test case 1: slice last dimension
        x1 = torch.randn(4, 8, 64, device="neuron", dtype=torch.float32)
        sliced_1 = x1[..., :32]

        if not sliced_1.is_contiguous():
            compile_called["count"] = 0
            result_1 = sliced_1.contiguous()
            assert (
                compile_called["count"] == 1
            ), f"Expected 1 compilation, got {compile_called['count']}"
            assert result_1.is_contiguous()
            assert torch.allclose(result_1.cpu(), sliced_1.cpu())

        # Test case 2: different axis - should compile new kernel
        x2 = torch.randn(8, 64, 32, device="neuron", dtype=torch.float32)
        sliced_2 = x2[:4, ...]

        if not sliced_2.is_contiguous():
            compile_called["count"] = 0
            result_2 = sliced_2.contiguous()
            assert (
                compile_called["count"] == 1
            ), f"Expected 1 compilation, got {compile_called['count']}"
            assert result_2.is_contiguous()
            assert torch.allclose(result_2.cpu(), sliced_2.cpu())
