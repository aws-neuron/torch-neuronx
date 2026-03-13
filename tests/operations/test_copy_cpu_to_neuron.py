import pytest
import torch

import torch_neuronx


def test_copy_cpu_to_neuron_success():
    """Test successful copy from CPU to Neuron"""
    cpu_tensor = torch.randn(4, 4)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    # Copy should succeed
    neuron_tensor.copy_(cpu_tensor)

    # Verify by copying back and checking data. Destination CPU tensor must be contiguous.
    cpu_result = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_result)


def test_copy_cpu_to_neuron_non_contiguous_success():
    """Test successful copy from non-contiguous CPU tensor to Neuron"""
    cpu_tensor = torch.randn(4, 4).transpose(0, 1)
    neuron_tensor = torch.empty(4, 4, device="neuron:0")

    assert not cpu_tensor.is_contiguous()

    # Copy should now succeed even if CPU tensor is non-contiguous
    neuron_tensor.copy_(cpu_tensor)

    # Verify by copying back and checking data. Destination CPU tensor must be contiguous.
    cpu_result = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype)
    cpu_result.copy_(neuron_tensor)

    torch.testing.assert_close(cpu_tensor, cpu_result)


class TestAsyncCopySafety:
    def test_nonblocking_h2d_source_modification(self):
        size = 1_000_000
        src = torch.randn(size, device="cpu")
        dst = torch.empty(size, device="neuron")
        expected = src.clone()

        dst.copy_(src, non_blocking=True)
        src.fill_(999.0)

        torch.neuron.synchronize()
        assert torch.allclose(dst.cpu(), expected, rtol=1e-5)

    def test_nonblocking_h2d_source_deletion(self):
        size = 1_000_000
        dst = torch.empty(size, device="neuron")

        def create_and_copy():
            src = torch.randn(size, device="cpu")
            expected = src.clone()
            dst.copy_(src, non_blocking=True)
            return expected

        expected = create_and_copy()
        torch.neuron.synchronize()
        assert torch.allclose(dst.cpu(), expected, rtol=1e-5)

    def test_nonblocking_h2d_multiple_concurrent(self):
        n_copies = 10
        size = 100_000

        srcs = [torch.randn(size, device="cpu") for _ in range(n_copies)]
        dsts = [torch.empty(size, device="neuron") for _ in range(n_copies)]
        expecteds = [src.clone() for src in srcs]

        for src, dst in zip(srcs, dsts, strict=False):
            dst.copy_(src, non_blocking=True)

        for src in srcs:
            src.fill_(999.0)

        torch.neuron.synchronize()

        for i, (dst, expected) in enumerate(zip(dsts, expecteds, strict=False)):
            assert torch.allclose(dst.cpu(), expected, rtol=1e-5), f"Copy {i} corrupted"

    def test_nonblocking_d2h_correctness(self):
        size = 1_000_000
        src = torch.randn(size, device="neuron")
        dst = torch.empty(size, device="cpu")

        dst.copy_(src, non_blocking=True)
        torch.neuron.synchronize()

        assert torch.allclose(dst, src.cpu(), rtol=1e-5)

    def test_blocking_h2d_correctness(self):
        size = 1_000_000
        src = torch.randn(size, device="cpu")
        dst = torch.empty(size, device="neuron")
        expected = src.clone()

        dst.copy_(src, non_blocking=False)
        src.fill_(999.0)

        assert torch.allclose(dst.cpu(), expected, rtol=1e-5)

    def test_empty_tensor_copy(self):
        src = torch.empty(0, device="cpu")
        dst = torch.empty(0, device="neuron")

        dst.copy_(src, non_blocking=True)
        torch.neuron.synchronize()

        assert dst.numel() == 0
