import os

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_raises


def test_copy_neuron_to_cpu_basic():
    """Test basic Neuron to CPU copy"""
    # Create neuron tensor with known data
    cpu_source = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    # First copy data to neuron tensor
    neuron_tensor.copy_(cpu_source)

    # Now copy back to CPU
    cpu_result = torch.empty_like(cpu_source)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_source, cpu_result)


def test_copy_neuron_to_cpu_non_contiguous_dst():
    """Test that non-contiguous CPU destination works"""
    # Create a neuron tensor with data
    cpu_source = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")
    neuron_tensor.copy_(cpu_source)

    # Create non-contiguous CPU tensor with matching size
    cpu_base = torch.empty(4, 4)
    cpu_tensor = cpu_base.transpose(0, 1)

    assert not cpu_tensor.is_contiguous()

    cpu_tensor.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_source)


class TestAsyncCopySafety:
    def test_nonblocking_d2h_pinned_destination(self):
        size = 1_000_000
        src = torch.randn(size, device="neuron")
        dst = torch.empty(size).pin_memory(device="neuron")

        dst.copy_(src, non_blocking=True)
        torch.neuron.synchronize()

        assert torch.allclose(dst, src.cpu(), rtol=1e-5)

    def test_nonblocking_d2h_pinned_destination_deletion(self):
        size = 1_000_000
        src = torch.randn(size, device="neuron")

        def create_and_copy():
            dst = torch.empty(size).pin_memory(device="neuron")
            dst.copy_(src, non_blocking=True)
            return None

        create_and_copy()
        torch.neuron.synchronize()

    def test_nonblocking_d2h_pinned_multiple_concurrent(self):
        n_copies = 10
        size = 100_000

        srcs = [torch.randn(size, device="neuron") for _ in range(n_copies)]
        dsts = [torch.empty(size).pin_memory(device="neuron") for _ in range(n_copies)]

        for src, dst in zip(srcs, dsts, strict=False):
            dst.copy_(src, non_blocking=True)

        torch.neuron.synchronize()

        for i, (src, dst) in enumerate(zip(srcs, dsts, strict=False)):
            assert torch.allclose(dst, src.cpu(), rtol=1e-5), f"Copy {i} corrupted"


class TestToCpuNonBlocking:
    """Verify that .to("cpu", non_blocking=True) allocates pinned memory"""

    def test_to_cpu_nonblocking_is_pinned(self):
        src = torch.randn(1024, device="neuron")
        dst = src.to("cpu", non_blocking=True)
        torch.neuron.synchronize()
        assert dst.is_pinned()

    def test_to_cpu_blocking_is_not_pinned(self):
        src = torch.randn(1024, device="neuron")
        dst = src.to("cpu")
        assert not dst.is_pinned()

    def test_to_cpu_nonblocking_sanity_correctness(self):
        cpu_orig = torch.randn(4, 4)
        src = cpu_orig.to("neuron")
        dst = src.to("cpu", non_blocking=True)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, cpu_orig)

    def test_to_cpu_nonblocking_with_sanity_dtype_cast(self):
        src = torch.randn(512, device="neuron", dtype=torch.float32)
        dst = src.to(device="cpu", dtype=torch.float16, non_blocking=True)
        expected = src.to("cpu").to(torch.float16)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, expected)


# ═══════════════════════════════════════════════════════════════
#  Conjugate / negation bit materialization (Neuron → CPU)
#
#  CopyNeuronToCPU must apply conj/neg metadata after the raw byte copy.
# ═══════════════════════════════════════════════════════════════


class TestConjBitNeuronToCpu:
    """Neuron→CPU copy must materialize the conjugate bit."""

    @pytest.mark.parametrize("cpp_dispatch", [False, True], ids=["python_dispatch", "cpp_dispatch"])
    def test_copy_conj_to_cpu(self, cpp_dispatch):
        os.environ["TORCH_NEURONX_CPP_DISPATCH"] = "1" if cpp_dispatch else "0"
        try:
            t = torch.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype=torch.complex64, device="neuron")
            dst = torch.empty(2, dtype=torch.complex64)
            dst.copy_(t.conj())
            expected = torch.tensor([1.0 - 2.0j, 3.0 + 4.0j], dtype=torch.complex64)
            torch.testing.assert_close(dst, expected)
        finally:
            os.environ.pop("TORCH_NEURONX_CPP_DISPATCH", None)

    def test_to_cpu_conj(self):
        t = torch.randn(16, dtype=torch.complex64)
        result = t.to("neuron").conj().cpu()
        torch.testing.assert_close(result, t.conj().resolve_conj())

    def test_to_cpu_neg(self):
        t = torch.randn(16, dtype=torch.complex64)
        result = t.to("neuron")._neg_view().cpu()
        torch.testing.assert_close(result, -t)

    def test_copy_conj_broadcast_to_cpu(self):
        """Conj'd Neuron tensor with broadcast into CPU destination."""
        src = torch.randn(1, 4, dtype=torch.complex64)
        src_neuron = src.to("neuron").conj()
        dst = torch.empty(3, 4, dtype=torch.complex64)
        dst.copy_(src_neuron)
        expected = src.conj().resolve_conj().expand(3, 4)
        torch.testing.assert_close(dst, expected)

    def test_copy_conj_and_neg_to_cpu(self):
        """Tensor with both conj and neg bits copied to CPU."""
        t = torch.randn(4, 4, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()._neg_view()
        result = t_neuron.clone().cpu()
        expected = -(t.conj().resolve_conj())
        torch.testing.assert_close(result, expected)

    def test_copy_conj_to_cpu_non_blocking(self):
        """Non-blocking Neuron→CPU copy with conj bit."""
        t = torch.randn(16, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()
        dst = torch.empty(16, dtype=torch.complex64).pin_memory("neuron")
        dst.copy_(t_neuron, non_blocking=True)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, t.conj().resolve_conj())

    def test_to_copy_conj_with_dtype_cast(self):
        """_to_copy path: conj'd Neuron tensor cast to different dtype on CPU."""
        t = torch.randn(8, dtype=torch.complex64)
        t_neuron = t.to("neuron").conj()
        result = t_neuron.to(dtype=torch.complex128, device="cpu")
        expected = t.conj().resolve_conj().to(torch.complex128)
        torch.testing.assert_close(result, expected)
