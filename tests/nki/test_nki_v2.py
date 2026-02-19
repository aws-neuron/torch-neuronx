"""Test NKI V2 functionality with nki library."""

import pytest
import torch

from torch_neuronx.nki_kernel import IS_NKI_V2_AVAILABLE

if IS_NKI_V2_AVAILABLE:
    import nki
    import nki.isa as nisa
    import nki.language as nl


@pytest.mark.skipif(not IS_NKI_V2_AVAILABLE, reason="NKI V2 (nki library) not available")
class TestNKIV2:
    """Test cases for NKI V2 functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.fixture
    def sample_tensors_2d(self, device):
        """Sample 2D tensors (required for NKI V2)."""
        x1 = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float32)
        x2 = torch.tensor([[3.0, 4.0]], device=device, dtype=torch.float32)
        y = torch.zeros((1, 2), device=device, dtype=torch.float32)
        return x1, x2, y

    def test_nki_v2_basic_kernel(self, sample_tensors_2d):
        """Test basic NKI V2 kernel functionality."""
        from torch_neuronx import nki_op, wrap_nki

        @nki.jit
        def add_kernel(x1, x2, y):
            x1_sbuf = nl.ndarray(x1.shape, x1.dtype, nl.sbuf)
            x2_sbuf = nl.ndarray(x2.shape, x2.dtype, nl.sbuf)
            y_sbuf = nl.ndarray(y.shape, y.dtype, nl.sbuf)

            nisa.dma_copy(x1_sbuf, x1)
            nisa.dma_copy(x2_sbuf, x2)
            nisa.tensor_tensor(y_sbuf, x1_sbuf, x2_sbuf, nl.add)
            nisa.dma_copy(y, y_sbuf)
            return y

        @nki_op("test::add_v2", mutates_args={"y"})
        def add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return wrap_nki(add_kernel)(x1, x2, y)

        x1, x2, y = sample_tensors_2d
        expected = x1 + x2

        # Call the kernel
        result = add(x1, x2, y)

        # Check that result has correct values
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)
        # Note: y mutation not tested as aliasing is not supported in NKI V2 yet

    def test_nki_v2_with_torch_compile(self, sample_tensors_2d):
        """Test NKI V2 kernel with torch.compile."""
        from torch_neuronx import nki_op, wrap_nki

        @nki.jit
        def mul_kernel(x1, x2, y):
            x1_sbuf = nl.ndarray(x1.shape, x1.dtype, nl.sbuf)
            x2_sbuf = nl.ndarray(x2.shape, x2.dtype, nl.sbuf)
            y_sbuf = nl.ndarray(y.shape, y.dtype, nl.sbuf)

            nisa.dma_copy(x1_sbuf, x1)
            nisa.dma_copy(x2_sbuf, x2)
            nisa.tensor_tensor(y_sbuf, x1_sbuf, x2_sbuf, nl.multiply)
            nisa.dma_copy(y, y_sbuf)
            return y

        @nki_op("test::mul_v2", mutates_args={"y"})
        def mul(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return wrap_nki(mul_kernel)(x1, x2, y)

        x1, x2, y = sample_tensors_2d
        expected = x1 * x2

        # Compile the kernel
        compiled_mul = torch.compile(lambda *args: mul(*args), backend="neuron", fullgraph=True)

        # Call the compiled kernel
        result = compiled_mul(x1, x2, y)

        # Check that result is correct
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)
        # Note: y mutation not tested as aliasing is not supported in NKI V2 yet

    def test_nki_v2_kernel_detection(self):
        """Test that NKI V2 kernels are detected correctly by wrap_nki."""
        from torch_neuronx import wrap_nki

        @nki.jit
        def dummy_kernel(x):
            return x

        # Check that the kernel's module is from nki (not neuronxcc)
        kernel_module = dummy_kernel.__class__.__module__
        assert "neuronxcc" not in kernel_module
        assert "nki" in kernel_module or "TraceKernel" in str(type(dummy_kernel))

        # wrap_nki should work without errors
        wrapped = wrap_nki(dummy_kernel)
        assert wrapped is not None

    def test_nki_v2_different_dtypes(self, device):
        """Test NKI V2 kernel with different data types."""
        from torch_neuronx import nki_op, wrap_nki

        @nki.jit
        def copy_kernel(x, y):
            x_sbuf = nl.ndarray(x.shape, x.dtype, nl.sbuf)
            y_sbuf = nl.ndarray(y.shape, y.dtype, nl.sbuf)

            nisa.dma_copy(x_sbuf, x)
            # Simple copy operation
            nisa.dma_copy(y_sbuf, x_sbuf)
            nisa.dma_copy(y, y_sbuf)
            return y

        @nki_op("test::copy_v2", mutates_args={"y"})
        def copy_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return wrap_nki(copy_kernel)(x, y)

        # Test with different dtypes
        for dtype in [torch.float32, torch.float16]:
            x = torch.tensor([[1.0, 2.0]], device=device, dtype=dtype)
            y = torch.zeros((1, 2), device=device, dtype=dtype)

            result = copy_op(x, y)

            # Check that result has correct values
            assert torch.allclose(result.cpu(), x.cpu(), atol=1e-5)
            assert result.dtype == dtype
            # Note: y mutation not tested as aliasing is not supported in NKI V2 yet

    def test_nki_kernel_opts(self, sample_tensors_2d):
        from nki._enums import OptLevel
        from nki.compiler.backends.neuron.CompileOpts import AllocatorType

        from torch_neuronx import nki_op, wrap_nki
        from torch_neuronx.nki_hop import kernel_registry

        @nki.jit(
            enable_const_rewrite=True,
            opt_level=OptLevel.skip_middle_end,
            allocator_type=AllocatorType.auto_spill,
        )
        def add_kernel(x1, x2):
            y = nl.ndarray(x1.shape, x1.dtype, nl.shared_hbm)
            x1_sbuf = nl.ndarray(x1.shape, x1.dtype, nl.sbuf)
            x2_sbuf = nl.ndarray(x2.shape, x2.dtype, nl.sbuf)
            y_sbuf = nl.ndarray(y.shape, y.dtype, nl.sbuf)

            nisa.dma_copy(x1_sbuf, x1)
            nisa.dma_copy(x2_sbuf, x2)
            nisa.tensor_tensor(y_sbuf, x1_sbuf, x2_sbuf, nl.add)
            nisa.dma_copy(y, y_sbuf)
            return y

        wrapped = wrap_nki(add_kernel)
        wrapped_kernel = kernel_registry.get_kernel(wrapped.kernel_idx)
        assert wrapped_kernel.opts.enable_const_rewrite
        assert wrapped_kernel.opts.opt_level == OptLevel.skip_middle_end
        assert wrapped_kernel.opts.allocator_type == AllocatorType.auto_spill

        @nki_op("test::kernel_opts_add", mutates_args={"y"})
        def add(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return wrap_nki(add_kernel)(x1, x2, y)

        x1, x2, y = sample_tensors_2d
        expected = x1 + x2

        # Call the kernel
        result = add(x1, x2, y)

        # Check that result has correct values
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
