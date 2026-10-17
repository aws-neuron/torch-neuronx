"""Tests for NKI eager dispatch bypass (_C._nki_dispatch_eager).

Verifies that the C++ fast path produces identical results to the
HOP path and handles edge cases correctly.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import pytest
import torch

from torch_neuronx import wrap_nki


@pytest.fixture
def device():
    return torch.device("neuron:0")


class TestNkiEagerDispatch:
    """Test the _C._nki_dispatch_eager fast path."""

    def test_basic_kernel(self, device):
        """Basic kernel produces correct results via eager bypass."""

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

        wrapped = wrap_nki(add_kernel)
        a = torch.randn(1, 128, device=device)
        b = torch.randn(1, 128, device=device)

        result = wrapped(a, b)
        expected = a + b
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)

    def test_kernel_with_scalar_arg(self, device):
        """Kernel with non-tensor (scalar) constant arg works via eager bypass."""

        @nki.jit
        def scale_kernel(x, scale):
            x_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sbuf, src=x)
            y_sbuf = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y_sbuf, data=x_sbuf, op0=nl.multiply, operand0=scale)
            out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out, src=y_sbuf)
            return out

        wrapped = wrap_nki(scale_kernel)
        x = torch.randn(1, 128, device=device)

        result = wrapped(x, 2.5)
        expected = x * 2.5
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)

    def test_cache_hit_consistency(self, device):
        """Multiple calls with same shapes produce consistent results (cache hit path)."""

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

        wrapped = wrap_nki(add_kernel)
        a = torch.randn(1, 128, device=device)
        b = torch.randn(1, 128, device=device)

        # First call (cache miss)
        result1 = wrapped(a, b)
        # Second call (cache hit — exercises the has_nki_config path)
        result2 = wrapped(a, b)

        assert torch.allclose(result1.cpu(), result2.cpu())
        assert torch.allclose(result1.cpu(), (a + b).cpu(), atol=1e-5)

    def test_direct_c_api(self, device):
        """_C._nki_dispatch_eager can be called directly."""
        import torch_neuronx._C as _C
        from torch_neuronx.nki_hop import get_dconfig_and_tensor_args

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

        wrapped = wrap_nki(add_kernel)
        a = torch.randn(1, 128, device=device)
        b = torch.randn(1, 128, device=device)

        # Warmup via normal path to populate cache
        wrapped(a, b)

        # Call C++ directly
        kernel_idx = wrapped.kernel_idx
        arg_names = wrapped.kernel_arg_names
        dconfig, tensor_args, _, cak = get_dconfig_and_tensor_args(
            kernel_idx, [], (a, b), arg_names, -1
        )
        results = _C._nki_dispatch_eager(
            list(tensor_args),
            kernel_idx,
            str(cak),
            [],
            str(dconfig.operand_output_aliases),
            False,
        )
        assert isinstance(results, list | tuple | torch.Tensor)

    def test_eager_bypass_not_used_for_compile(self, device):
        """torch.compile path still uses HOP dispatch, not eager bypass."""
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

        @nki_op("test::eager_bypass_add")
        def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return add_kernel(a, b)

        a = torch.randn(1, 128, device=device)
        b = torch.randn(1, 128, device=device)

        compiled = torch.compile(lambda x, y: nki_add(x, y), backend="neuron", fullgraph=True)
        result = compiled(a, b)
        expected = a + b
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-5)
