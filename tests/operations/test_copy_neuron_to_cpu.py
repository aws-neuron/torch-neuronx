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
        assert dst.is_pinned(device="neuron")

    def test_to_cpu_blocking_is_not_pinned(self):
        src = torch.randn(1024, device="neuron")
        dst = src.to("cpu")
        assert not dst.is_pinned(device="neuron")

    def test_to_cpu_nonblocking_sanity_correctness(self):
        cpu_orig = torch.randn(4, 4)
        src = cpu_orig.to("neuron")
        dst = src.to("cpu", non_blocking=True)
        torch.neuron.synchronize()
        torch.testing.assert_close(dst, cpu_orig)

    def test_to_cpu_nonblocking_with_sanity_dtype_cast(self):
        src = torch.randn(512, device="neuron", dtype=torch.float32)
        dst = src.to(device="cpu", dtype=torch.float16, non_blocking=True)
        torch.neuron.synchronize()
        expected = src.to("cpu").to(torch.float16)
        torch.testing.assert_close(dst, expected)
