"""Integration tests for packed int64 execution on Neuron device.

Requires TORCH_NEURONX_INT64_MODE=packed and a Neuron device.
Tests the full round-trip: transfer, packed view extraction, compute, materialization.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_did_not_run_on_neuron,
    assert_op_runs_on_neuron,
    track_neuron_ops,
)
from torch_neuronx.packed_storage import (
    materialize_int64_from_i32,
    packed_int64_execution_view,
    packed_physical_view,
)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPackedInt64ExecutionView:
    """Tests for packed int64 view extraction and materialization."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    def test_execution_view_contiguous_round_trip(self, device):
        """Values within int32 range survive extract→materialize round trip."""
        cpu_vals = torch.tensor([1, -1, 42, 0, 2**31 - 1, -(2**31)], dtype=torch.int64)
        dev = cpu_vals.to(device)

        low = packed_int64_execution_view(dev, contiguous=True)
        assert low.dtype == torch.int32
        assert low.shape == dev.shape
        assert low.is_contiguous()

        result = torch.empty_like(dev)
        materialize_int64_from_i32(result, low)

        expected = cpu_vals.to(torch.int32).to(torch.int64)
        torch.testing.assert_close(result.cpu(), expected)

    def test_execution_view_preserves_shape(self, device):
        """Multi-dimensional tensors preserve logical shape."""
        dev = torch.arange(12, dtype=torch.int64, device=device).reshape(3, 4)

        low = packed_int64_execution_view(dev)
        assert low.shape == (3, 4)
        assert low.dtype == torch.int32

    def test_materialize_sign_extends(self, device):
        """Negative int32 values sign-extend to negative int64."""
        low_words = torch.tensor([-1, -100, 0, 100], dtype=torch.int32, device=device)
        dst = torch.empty(4, dtype=torch.int64, device=device)

        materialize_int64_from_i32(dst, low_words)

        expected = torch.tensor([-1, -100, 0, 100], dtype=torch.int64)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_materialize_zero_extends(self, device):
        """With zero_extend=True, negative int32 becomes large positive int64."""
        low_words = torch.tensor([-1], dtype=torch.int32, device=device)
        dst = torch.empty(1, dtype=torch.int64, device=device)

        materialize_int64_from_i32(dst, low_words, zero_extend=True)

        expected = torch.tensor([0xFFFFFFFF], dtype=torch.int64)
        torch.testing.assert_close(dst.cpu(), expected)

    def test_physical_view_on_device(self, device):
        """packed_physical_view creates correct int32 view on Neuron."""
        dev = torch.tensor([7], dtype=torch.int64, device=device)
        phys = packed_physical_view(dev)

        assert phys.dtype == torch.int32
        assert phys.shape == (1, 2)
        cpu_phys = phys.cpu()
        assert int(cpu_phys[0, 0]) == 7
        assert int(cpu_phys[0, 1]) == 0


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestPackedInt64NoFallback:
    """Verify int64 ops execute on Neuron without CPU fallback in packed mode."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    def test_add_int64_no_fallback(self, device):
        """int64 add should run on Neuron, not fall back to CPU."""
        with track_neuron_ops():
            a = torch.tensor([1, -5, 100], dtype=torch.int64, device=device)
            b = torch.tensor([10, 3, -50], dtype=torch.int64, device=device)
            c = a + b

            assert_op_did_not_run_on_neuron("aten::neuron_to_cpu")
            expected = torch.tensor([11, -2, 50], dtype=torch.int64)
            torch.testing.assert_close(c.cpu(), expected)
            assert_op_runs_on_neuron("aten::add")

    def test_mul_int64_no_fallback(self, device):
        """int64 mul should run on Neuron, not fall back to CPU."""
        with track_neuron_ops():
            a = torch.tensor([2, -3, 7], dtype=torch.int64, device=device)
            b = torch.tensor([5, 4, -2], dtype=torch.int64, device=device)
            c = a * b

            assert_op_did_not_run_on_neuron("aten::neuron_to_cpu")
            expected = torch.tensor([10, -12, -14], dtype=torch.int64)
            torch.testing.assert_close(c.cpu(), expected)
            assert_op_runs_on_neuron("aten::mul")
