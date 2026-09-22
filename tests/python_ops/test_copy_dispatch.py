"""Tests for aten::copy_ via C++ DispatchCopy.

Covers CPU↔Neuron transfers, Neuron→Neuron same-dtype (NRT),
Neuron→Neuron unsupported-dtype (CPU bounce), Neuron→Neuron
supported-dtype (MLIR), empty tensors, self-copy, broadcasting,
scalar src, and non-contiguous dst.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)


class TestCopyDispatch:
    """Test suite for aten::copy_ via C++ DispatchCopy."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    # ── CPU → Neuron ──

    def test_copy_cpu_to_neuron_same_dtype(self, device):
        """CPU fp32 → Neuron fp32."""
        src = torch.randn(4, 4)
        dst = torch.empty(4, 4, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst.cpu(), src)

    def test_copy_cpu_to_neuron_dtype_cast(self, device):
        """CPU fp64 → Neuron fp32 dst (dtype cast on CPU before transfer)."""
        src = torch.randn(4, 4, dtype=torch.float64)
        dst = torch.empty(4, 4, dtype=torch.float32, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst.cpu(), src.float())

    # ── Neuron → CPU ──

    def test_copy_neuron_to_cpu_same_dtype(self, device):
        """Neuron fp32 → CPU fp32."""
        src_cpu = torch.randn(4, 4)
        src = src_cpu.to(device)
        dst = torch.empty(4, 4)
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst, src_cpu)

    def test_copy_neuron_to_cpu_dtype_cast(self, device):
        """Neuron fp32 → CPU fp64 (dtype cast on CPU after transfer)."""
        src_cpu = torch.randn(4, 4, dtype=torch.float32)
        src = src_cpu.to(device)
        dst = torch.empty(4, 4, dtype=torch.float64)
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst, src_cpu.double())

    # ── Neuron → Neuron ──

    def test_copy_neuron_to_neuron_same_dtype(self, device):
        """Neuron fp32 → Neuron fp32 (NRT D2D)."""
        src_cpu = torch.randn(4, 4)
        src = src_cpu.to(device)
        dst = torch.empty(4, 4, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst.cpu(), src_cpu)

    def test_copy_neuron_to_neuron_unsupported_dtype(self, device):
        """Neuron fp64 → Neuron fp64 (CPU bounce for unsupported dtype)."""
        src_cpu = torch.randn(4, 4, dtype=torch.float64)
        src = src_cpu.to(device)
        dst = torch.empty(4, 4, dtype=torch.float64, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst.cpu(), src_cpu)

    def test_copy_neuron_to_neuron_bool(self, device):
        """Neuron bool → Neuron bool (CPU bounce for unsupported dtype)."""
        src_cpu = torch.randint(0, 2, (4, 4)).bool()
        src = src_cpu.to(device)
        dst = torch.empty(4, 4, dtype=torch.bool, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        assert torch.equal(dst.cpu(), src_cpu)

    def test_copy_neuron_to_neuron_mlir_cast(self, device):
        """Neuron fp32 → Neuron bf16 (MLIR lowering path)."""
        src_cpu = torch.randn(4, 4, dtype=torch.float32)
        src = src_cpu.to(device)
        dst = torch.empty(4, 4, dtype=torch.bfloat16, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        expected = src_cpu.bfloat16()
        torch.testing.assert_close(dst.cpu(), expected)

    # ── Edge cases ──

    def test_copy_empty_tensor(self, device):
        """Empty tensor copy is a no-op."""
        src = torch.empty(0, dtype=torch.float32)
        dst = torch.empty(0, dtype=torch.float32, device=device)
        dst.copy_(src)
        assert dst.numel() == 0

    def test_copy_self_copy(self, device):
        """Copy tensor to itself is a no-op."""
        src_cpu = torch.randn(4, 4)
        t = src_cpu.to(device)
        t.copy_(t)
        torch_neuronx.synchronize()
        torch.testing.assert_close(t.cpu(), src_cpu)

    def test_copy_broadcast(self, device):
        """Broadcast (1,4) src into (3,4) dst."""
        src_cpu = torch.randn(1, 4)
        src = src_cpu.to(device)
        dst = torch.empty(3, 4, device=device)
        dst.copy_(src)
        torch_neuronx.synchronize()
        expected = src_cpu.expand(3, 4).contiguous()
        torch.testing.assert_close(dst.cpu(), expected)

    def test_copy_non_contiguous_dst(self, device):
        """Copy into a non-contiguous (transposed) dst."""
        src_cpu = torch.randn(4, 3)
        src = src_cpu.to(device)
        # Create non-contiguous dst via transpose
        backing = torch.empty(3, 4, device=device)
        dst = backing.t()  # shape (4,3) but non-contiguous
        assert not dst.is_contiguous()
        dst.copy_(src)
        torch_neuronx.synchronize()
        torch.testing.assert_close(dst.cpu(), src_cpu)

    # ── Op tracking ──

    def test_copy_runs_on_neuron(self, device):
        """Verify copy_ is tracked as running on Neuron."""
        with track_neuron_ops():
            src = torch.randn(4, 4)
            dst = torch.empty(4, 4, device=device)
            dst.copy_(src)
            torch_neuronx.synchronize()
            assert_op_runs_on_neuron("aten::copy_")
