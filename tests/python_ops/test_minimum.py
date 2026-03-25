"""Tests for aten::minimum and aten::minimum.out on Neuron.

These tests verify that:
  1. aten::minimum executes on Neuron hardware (not CPU fallback).
  2. aten::minimum.out executes on Neuron and writes the result into the
     pre-allocated out tensor (aliasing requirement).
  3. Both ops produce numerically correct results across dtypes/shapes.
"""

import pytest
import torch

try:  # pragma: no cover - environment-dependent import
    import torch_neuronx
except ImportError:
    pytest.skip(
        "torch_neuronx C++ extension is not built; skipping Neuron minimum tests",
        allow_module_level=True,
    )

from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMinimum:
    """Tests for aten::minimum and aten::minimum.out."""

    # ------------------------------------------------------------------
    # aten::minimum
    # ------------------------------------------------------------------

    def test_minimum_basic(self):
        """Element-wise minimum of two flat float32 tensors."""
        with track_neuron_ops():
            cpu_a = torch.tensor([1.0, 5.0, -3.0, 0.0], dtype=torch.float32)
            cpu_b = torch.tensor([0.0, 2.0, -4.0, 10.0], dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)
            result = torch.minimum(neuron_a, neuron_b)

            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum")

    def test_minimum_bfloat16(self):
        """minimum with bfloat16 — the primary training dtype on Trainium."""
        with track_neuron_ops():
            cpu_a = torch.tensor([1.5, -2.0, 0.0, 3.75], dtype=torch.bfloat16)
            cpu_b = torch.tensor([1.0, -1.0, 0.5, 5.0], dtype=torch.bfloat16)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)
            result = torch.minimum(neuron_a, neuron_b)

            torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
            assert_op_runs_on_neuron("aten::minimum")

    def test_minimum_broadcasting(self):
        """minimum with shapes that require broadcasting."""
        with track_neuron_ops():
            # [2, 3] vs [3] — broadcast over rows
            cpu_a = torch.tensor([[1.0, 5.0, -3.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
            cpu_b = torch.tensor([0.0, 3.0, -4.0], dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)
            result = torch.minimum(neuron_a, neuron_b)

            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum")

    def test_minimum_mixed_dtypes(self):
        """minimum promotes dtypes when inputs differ (float32 + bfloat16)."""
        with track_neuron_ops():
            cpu_a = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
            cpu_b = torch.tensor([0.5, -3.0, 4.0], dtype=torch.bfloat16)

            # CPU reference uses the same promotion
            promoted = torch.result_type(cpu_a, cpu_b)
            expected = torch.minimum(cpu_a.to(promoted), cpu_b.to(promoted))

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            result = torch.minimum(neuron_a, neuron_b)
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
            assert_op_runs_on_neuron("aten::minimum")

    def test_minimum_negative_values(self):
        """minimum correctly handles all-negative inputs."""
        with track_neuron_ops():
            cpu_a = torch.tensor([-1.0, -5.0, -3.0, -0.5], dtype=torch.float32)
            cpu_b = torch.tensor([-2.0, -4.0, -4.0, -0.1], dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)
            result = torch.minimum(neuron_a, neuron_b)

            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum")

    def test_minimum_2d(self):
        """minimum on 2-D tensors."""
        with track_neuron_ops():
            cpu_a = torch.randn(4, 8, dtype=torch.float32)
            cpu_b = torch.randn(4, 8, dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)
            result = torch.minimum(neuron_a, neuron_b)

            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum")

    # ------------------------------------------------------------------
    # aten::minimum.out  (the operator identified as missing by the
    # Launchpad finetuning project – appears during GRPO backward pass)
    # ------------------------------------------------------------------

    def test_minimum_out_writes_result(self):
        """aten::minimum.out must write into the provided out tensor and return it."""
        with track_neuron_ops():
            cpu_a = torch.tensor([1.0, 5.0, -3.0, 0.0], dtype=torch.float32)
            cpu_b = torch.tensor([0.0, 2.0, -4.0, 10.0], dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)

            out = torch.empty_like(neuron_a)
            result = torch.minimum(neuron_a, neuron_b, out=out)

            # The .out convention: the returned tensor MUST alias out
            assert result is out, "aten::minimum.out must return the out tensor"
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum.out")

    def test_minimum_out_bfloat16(self):
        """aten::minimum.out with bfloat16 — common training path."""
        with track_neuron_ops():
            cpu_a = torch.tensor([1.5, -2.0, 0.0, 3.75], dtype=torch.bfloat16)
            cpu_b = torch.tensor([1.0, -1.0, 0.5, 5.0], dtype=torch.bfloat16)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)

            out = torch.empty_like(neuron_a)
            result = torch.minimum(neuron_a, neuron_b, out=out)

            assert result is out
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=1e-2)
            assert_op_runs_on_neuron("aten::minimum.out")

    def test_minimum_out_2d(self):
        """aten::minimum.out on 2-D tensors."""
        with track_neuron_ops():
            cpu_a = torch.randn(4, 8, dtype=torch.float32)
            cpu_b = torch.randn(4, 8, dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")

            expected = torch.minimum(cpu_a, cpu_b)

            out = torch.empty_like(neuron_a)
            result = torch.minimum(neuron_a, neuron_b, out=out)

            assert result is out
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum.out")

    def test_minimum_out_does_not_cpu_fallback(self):
        """Confirm aten::minimum.out is NOT falling back to CPU."""
        with track_neuron_ops():
            cpu_a = torch.tensor([1.0, 5.0, -3.0, 0.0], dtype=torch.float32)
            cpu_b = torch.tensor([0.0, 2.0, -4.0, 10.0], dtype=torch.float32)

            neuron_a = cpu_a.to("neuron")
            neuron_b = cpu_b.to("neuron")
            out = torch.empty_like(neuron_a)

            torch.minimum(neuron_a, neuron_b, out=out)

            fallback_ops = torch_neuronx.get_fallback_ops()
            assert "aten::minimum.out" not in fallback_ops, (
                f"aten::minimum.out fell back to CPU. Fallback ops: {fallback_ops}"
            )

    # ------------------------------------------------------------------
    # Integration: minimum inside GRPO surrogate loss
    # (torch.minimum(unclipped, clipped) is the exact call site in grpo.py)
    # ------------------------------------------------------------------

    def test_minimum_in_grpo_surrogate(self):
        """Simulate the GRPO surrogate: torch.minimum(unclipped, clipped)."""
        with track_neuron_ops():
            batch = (2, 4)
            unclipped_cpu = torch.randn(*batch, dtype=torch.float32)
            clipped_cpu = torch.clamp(unclipped_cpu, -0.2, 0.2)

            unclipped_neuron = unclipped_cpu.to("neuron")
            clipped_neuron = clipped_cpu.to("neuron")

            expected = torch.minimum(unclipped_cpu, clipped_cpu)
            result = torch.minimum(unclipped_neuron, clipped_neuron)

            torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-4)
            assert_op_runs_on_neuron("aten::minimum")
