"""Test NKI V3 kernels in eager, nki_op, and torch.compile modes."""

import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch


class TestNKIV3:
    """Test cases for NKI V3 functionality."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    @pytest.fixture
    def sample_tensors(self, device):
        """Pair of 2-D float32 tensors on neuron."""
        a = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float32)
        b = torch.tensor([[3.0, 4.0]], device=device, dtype=torch.float32)
        return a, b

    def test_nki_v3_basic_kernel(self, sample_tensors):
        """Test basic NKI V3 kernel functionality."""
        from torch_neuronx import nki_op

        @nki.jit
        def add_kernel(a, b):
            sa = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sa, src=a)

            sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=b)

            sc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)

            out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sc)
            return out

        @nki_op("test::add_v3")
        def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return add_kernel(a, b)

        a, b = sample_tensors
        expected = a + b

        result = add(a, b)

        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)

    def test_nki_op_with_torch_compile(self, sample_tensors):
        """Test NKI V3 kernel with torch.compile."""
        from torch_neuronx import nki_op

        @nki.jit
        def mul_kernel(a, b):
            sa = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sa, src=a)

            sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=b)

            sc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.multiply)

            out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sc)
            return out

        @nki_op("test::mul_v3")
        def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return mul_kernel(a, b)

        a, b = sample_tensors
        expected = a * b

        compiled_mul = torch.compile(lambda *args: mul(*args), backend="neuron", fullgraph=True)

        result = compiled_mul(a, b)

        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)

    def test_nki_v3_with_torch_compile(self, sample_tensors):
        """Test NKI V3 kernel with torch.compile."""
        from torch_neuronx import nki_op

        @nki.jit
        def mul_kernel(a, b):
            sa = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sa, src=a)

            sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=b)

            sc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.multiply)

            out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sc)
            return out

        @torch.compile(backend="neuron", fullgraph=True)
        def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return mul_kernel(a, b)

        a, b = sample_tensors
        expected = a * b

        result = mul(a, b)

        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)

    def test_nki_v3_kernel_detection(self):
        """Test that NKI V3 kernels are detected correctly."""
        from nki.framework.kernel import Kernel

        from torch_neuronx import wrap_nki

        @nki.jit
        def dummy_kernel(x):
            return x

        # V3 kernels should be Kernel instances
        assert isinstance(dummy_kernel, Kernel)

        # wrap_nki should work without errors
        wrapped = wrap_nki(dummy_kernel)
        assert wrapped is not None

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_nki_v3_different_dtypes(self, device, dtype):
        """Test NKI V3 kernel with different data types."""
        from torch_neuronx import nki_op

        @nki.jit
        def copy_kernel(x):
            x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sbuf, src=x)

            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=x_sbuf)
            return out

        @nki_op("test::copy_v3")
        def copy_op(x: torch.Tensor) -> torch.Tensor:
            return copy_kernel(x)

        x = torch.tensor([[1.0, 2.0]], device=device, dtype=dtype)

        result = copy_op(x)

        assert torch.allclose(result.cpu(), x.cpu(), atol=1e-5)
        assert result.dtype == dtype

    # ── eager (direct kernel call) ──────────────────────────────────────────
    def test_eager(self, sample_tensors):
        """Call the kernel directly (eager execution)."""

        @nki.jit
        def add_kernel(a, b):
            sa = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sa, src=a)

            sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sb, src=b)

            sc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)

            out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=sc)
            return out

        a, b = sample_tensors
        result = add_kernel(a, b)

        expected = a + b
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
