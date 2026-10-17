"""
Tests for int64 S64 HLO support and NRT reinterpretation.

Verifies int64 generates S64 in StableHLO instead of bouncing through CPU with int32, and
is properly reinterpreted by C++ backend NRT layer with no CPU bounces.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import count_cpu_bounce_calls
from torch_neuronx.utils import is_native_int64_enabled


@pytest.mark.skipif(
    not is_native_int64_enabled(),
    reason="Tests require native int64 passthrough (TORCH_NEURONX_DISABLE_NATIVE_INT64=1)",
)
class TestInt64NoBounceCount:
    """Test that int64 ops have zero CPU bounces."""

    def test_int64_add_operation_zero_bounces(self):
        """Test that int64 add has 0 CPU bounces."""
        x = torch.arange(100, dtype=torch.int64, device="neuron")
        y = torch.arange(100, dtype=torch.int64, device="neuron")
        torch.neuron.synchronize()

        def do_add():
            result = x + y
            torch.neuron.synchronize()
            return result

        counts = count_cpu_bounce_calls(do_add)

        assert counts["reads"] == 0, (
            f"int64 add: expected 0 reads, got {counts['reads']}. "
            "MLIR should use S64 HLO natively without bouncing to CPU."
        )
        assert counts["writes"] == 0, f"int64 add: expected 0 writes, got {counts['writes']}."

    def test_int64_where_operation_zero_bounces(self):
        """Test that torch.where with int64 has 0 CPU bounces."""
        condition = torch.tensor([True, False, True, False], device="neuron")
        x = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        y = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")
        torch.neuron.synchronize()

        def do_where():
            result = torch.where(condition, x, y)
            torch.neuron.synchronize()
            return result

        counts = count_cpu_bounce_calls(do_where)

        assert counts["reads"] == 0, f"where op: expected 0 reads, got {counts['reads']}"
        assert counts["writes"] == 0, f"where op: expected 0 writes, got {counts['writes']}"

    def test_torch_compile_int64_add_zero_bounces(self):
        """Test that torch.compile with int64 has 0 CPU bounces."""

        @torch.compile(backend="neuron")
        def add_fn(x, y):
            return x + y

        x = torch.arange(100, dtype=torch.int64, device="neuron")
        y = torch.arange(100, dtype=torch.int64, device="neuron")

        def do_compiled_add():
            return add_fn(x, y)

        counts = count_cpu_bounce_calls(do_compiled_add)

        assert counts["reads"] == 0, f"torch.compile: expected 0 reads, got {counts['reads']}"
        assert counts["writes"] == 0, f"torch.compile: expected 0 writes, got {counts['writes']}"

    def test_torch_compile_int64_where_zero_bounces(self):
        """Test that torch.compile with torch.where and int64 has 0 CPU bounces."""

        @torch.compile(backend="neuron")
        def where_fn(cond, x, y):
            return torch.where(cond, x, y)

        condition = torch.tensor([True, False, True, False], device="neuron")
        x = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="neuron")
        y = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="neuron")

        def do_compiled_where():
            return where_fn(condition, x, y)

        counts = count_cpu_bounce_calls(do_compiled_where)

        assert counts["reads"] == 0, f"torch.compile: expected 0 reads, got {counts['reads']}"
        assert counts["writes"] == 0, f"torch.compile: expected 0 writes, got {counts['writes']}"

    def test_multiple_int64_ops_zero_bounces(self):
        """Test that multiple int64 tensor-tensor ops don't accumulate bounces.

        Note: Uses tensor+tensor ops to isolate int64 S64 HLO fix from scalar
        conversion overhead (scalar ops trigger CPU→Neuron copy per scalar).
        """
        x = torch.arange(100, dtype=torch.int64, device="neuron")
        y = torch.arange(100, dtype=torch.int64, device="neuron")
        torch.neuron.synchronize()

        def do_multiple_ops():
            result = x
            for _ in range(10):
                result = result + y
            torch.neuron.synchronize()
            return result

        counts = count_cpu_bounce_calls(do_multiple_ops)

        assert counts["reads"] == 0, f"Multiple ops: expected 0 reads, got {counts['reads']}"
        assert counts["writes"] == 0, f"Multiple ops: expected 0 writes, got {counts['writes']}"
