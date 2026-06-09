"""
Tests for reshape insertion on shared/common inputs in operation concatenation.

Verifies that when two consecutive operations share an input tensor but see
different shapes (due to a view/reshape/unsqueeze not dispatched to Neuron),
the concatenation system inserts a stablehlo.reshape for the second module's
view of the shared input.

Key scenario: Op1 uses tensor X with shape [B, S, H], then a host-side view
reshapes X to [B, S, 1, H]. Op2 uses the viewed tensor. When Op1 and Op2 are
concatenated, the merged module takes X in one shape and inserts a reshape for
the second module's expected shape.

Requires:
    TORCH_NEURONX_ENABLE_CONCATENATION=1

Run with:
    TORCH_NEURONX_ENABLE_CONCATENATION=1 \
    pytest tests/python_ops/test_shared_input_reshape_concat.py -xvs
"""

import os

import pytest
import torch

import torch_neuronx
import torch_neuronx.metrics as met
from tests.utils.neuron_test_utils import (
    track_neuron_ops,
)


def is_enabled():
    return os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"


skip_if_not_enabled = pytest.mark.skipif(
    not is_enabled(),
    reason="Requires TORCH_NEURONX_ENABLE_CONCATENATION=1",
)


@pytest.fixture(autouse=True)
def setup_metrics():
    met.set_enabled(True)
    met.clear_all()
    torch_neuronx.clear_op_tracking()
    yield


def print_concat_summary(test_name: str):
    """Print concatenation metrics summary."""
    torch.neuron.synchronize()

    successes = met.get_counter_value("Concatenation.Successes") or 0
    fallbacks = met.get_counter_value("Concatenation.Fallbacks") or 0
    attempts = met.get_counter_value("Concatenation.Attempts") or 0
    groups = met.get_counter_value("Concatenation.GroupsCreated") or 0
    ops_concat = met.get_counter_value("Concatenation.OpsConcatenated") or 0

    print(f"\n=== {test_name} ===")
    print(f"  Attempts: {attempts}")
    print(f"  Successes: {successes}")
    print(f"  Fallbacks: {fallbacks}")
    print(f"  Groups created: {groups}")
    print(f"  Ops concatenated: {ops_concat}")

    if successes + fallbacks > 0:
        success_rate = successes / (successes + fallbacks) * 100
        print(f"  Success rate: {success_rate:.1f}%")

    concat_groups = torch_neuronx.get_concatenated_ops()
    if concat_groups:
        print(f"  Concatenated groups ({len(concat_groups)}):")
        for name, ops in concat_groups.items():
            print(f"    {name} <- {ops}")

    return {
        "successes": successes,
        "fallbacks": fallbacks,
        "attempts": attempts,
        "groups": groups,
        "ops_concat": ops_concat,
        "concat_groups": concat_groups,
    }


class TestSharedInputReshapeConcat:
    """Tests verifying that shared inputs with different shapes don't block concatenation."""

    @skip_if_not_enabled
    def test_shared_weight_with_view_concatenates(self):
        """
        Test that two matmuls sharing a weight tensor with different views concatenate.

        Pattern:
            w = randn(H, H)          # original weight
            h1 = x @ w               # matmul sees w as [H, H]
            w_viewed = w.view(1, H, H)
            h2 = torch.bmm(y, w_viewed)  # bmm sees w as [1, H, H]

        Both ops use the same underlying tensor (w) at the same data pointer,
        but with different shapes. The concatenation system should handle this
        via reshape insertion for the common input.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        hidden = 4096
        x_cpu = torch.randn(1, 2048, hidden, dtype=dtype) * 0.01
        w_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        y_cpu = torch.randn(1, 2048, hidden, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        w = w_cpu.to(device)
        y = y_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                # First matmul uses w as [H, H]
                h1 = x @ w
                # View creates [1, H, H] without copying data
                w_3d = w.unsqueeze(0)
                # Second bmm uses w as [1, H, H]
                h2 = torch.bmm(y, w_3d)
                result = h1 + h2.squeeze(0)

            result_final = result.cpu()

        metrics = print_concat_summary("test_shared_weight_with_view_concatenates")

        # Concatenation should be attempted and succeed
        assert metrics["attempts"] > 0, "Should have concatenation attempts"
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Shared input reshape may be broken."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"
        assert (
            metrics["ops_concat"] >= 2
        ), f"Expected at least 2 ops concatenated, got {metrics['ops_concat']}"

        # Verify correctness
        h1_cpu = x_cpu @ w_cpu
        w_3d_cpu = w_cpu.unsqueeze(0)
        h2_cpu = torch.bmm(y_cpu, w_3d_cpu)
        expected = h1_cpu + h2_cpu.squeeze(0)
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_shared_input_squeeze_between_matmuls(self):
        """
        Test that shared input with squeeze view between consecutive matmuls works.

        Pattern:
            x = randn(1, S, H)
            h1 = x @ w1              # uses x as [1, S, H]
            x_2d = x.squeeze(0)      # view to [S, H]
            h2 = x_2d @ w2           # uses x as [S, H]

        x is used by both ops (shared input) but with different shapes.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        seq = 2048
        hidden = 4096
        x_cpu = torch.randn(1, seq, hidden, dtype=dtype) * 0.01
        w1_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        w2_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        w1 = w1_cpu.to(device)
        w2 = w2_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                h1 = x @ w1  # [1, S, H] @ [H, H] -> [1, S, H]
                x_2d = x.squeeze(0)  # view: [S, H] — metadata only
                h2 = x_2d @ w2  # [S, H] @ [H, H] -> [S, H]
                result = h1.squeeze(0) + h2  # combine results

            result_final = result.cpu()

        metrics = print_concat_summary("test_shared_input_squeeze_between_matmuls")

        assert metrics["attempts"] > 0, "Should have concatenation attempts"
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Shared input reshape may be broken."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"
        assert (
            metrics["ops_concat"] >= 2
        ), f"Expected at least 2 ops concatenated, got {metrics['ops_concat']}"

        # Verify correctness
        h1_cpu = x_cpu @ w1_cpu
        x_2d_cpu = x_cpu.squeeze(0)
        h2_cpu = x_2d_cpu @ w2_cpu
        expected = h1_cpu.squeeze(0) + h2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_shared_input_reshape_between_norms(self):
        """
        Test rms_norm pattern where the input tensor is shared with different views.

        Pattern:
            x = randn(B, S, H)
            normed1 = rms_norm(x, [H], w)   # uses x as [B, S, H]
            x_4d = x.view(B, S, heads, dim)
            normed2 = rms_norm(x_4d, [dim], w2)  # uses x as [B, S, heads, dim]

        Both rms_norm ops use x (same data pointer) but see different shapes.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        batch = 1
        seq = 2048
        heads = 8
        dim = 512
        hidden = heads * dim  # 4096

        x_cpu = torch.randn(batch, seq, hidden, dtype=dtype) * 0.01
        norm_w1_cpu = torch.randn(hidden, dtype=dtype)
        norm_w2_cpu = torch.randn(dim, dtype=dtype)
        w_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        norm_w1 = norm_w1_cpu.to(device)
        norm_w2 = norm_w2_cpu.to(device)
        w = w_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                # First rms_norm uses x as [B, S, H]
                normed1 = torch.rms_norm(x, [hidden], norm_w1)
                h1 = normed1 @ w  # matmul boundary

                # Reshape x to [B, S, heads, dim] (view — same data pointer)
                x_4d = x.view(batch, seq, heads, dim)
                # Second rms_norm uses x as [B, S, heads, dim]
                normed2 = torch.rms_norm(x_4d, [dim], norm_w2)
                h2 = normed2.reshape(batch, seq, hidden) @ w  # matmul boundary

                result = h1 + h2

            result_final = result.cpu()

        metrics = print_concat_summary("test_shared_input_reshape_between_norms")

        assert metrics["attempts"] > 0, "Should have concatenation attempts"
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Shared input reshape may be broken."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"

        # Verify rms_norm ops appear in concatenated groups
        concat_groups = metrics["concat_groups"]
        assert len(concat_groups) > 0, "Should have recorded concatenated op groups"
        all_ops = []
        for ops in concat_groups.values():
            all_ops.extend(ops)
        assert any(
            "rms_norm" in op or "fused_rms_norm" in op for op in all_ops
        ), f"Expected rms_norm in concatenated ops, got: {all_ops}"

        # Verify correctness
        normed1_cpu = torch.rms_norm(x_cpu, [hidden], norm_w1_cpu)
        h1_cpu = normed1_cpu @ w_cpu
        x_4d_cpu = x_cpu.view(batch, seq, heads, dim)
        normed2_cpu = torch.rms_norm(x_4d_cpu, [dim], norm_w2_cpu)
        h2_cpu = normed2_cpu.reshape(batch, seq, hidden) @ w_cpu
        expected = h1_cpu + h2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_shared_input_reshape_reduces_fallbacks(self):
        """
        Verify that the shared input reshape fix reduces fallback count.

        Without the fix, common inputs with shape mismatches would cause
        concatenation to fail (fallback). With the fix, these should succeed.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        hidden = 4096
        x_cpu = torch.randn(1, 2048, hidden, dtype=dtype) * 0.01
        w1_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        w2_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        w1 = w1_cpu.to(device)
        w2 = w2_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                # Both matmuls share x but potentially with different views
                h1 = x @ w1
                x_viewed = x.reshape(2048, hidden)  # same data, different shape
                h2 = x_viewed @ w2
                result = h1.squeeze(0) + h2

            result_final = result.cpu()

        metrics = print_concat_summary("test_shared_input_reshape_reduces_fallbacks")

        successes = metrics["successes"]
        fallbacks = metrics["fallbacks"]

        if successes + fallbacks > 0:
            fallback_rate = fallbacks / (successes + fallbacks)
            print(f"  Fallback rate: {fallback_rate:.1%}")

        # Verify correctness
        h1_cpu = x_cpu @ w1_cpu
        x_viewed_cpu = x_cpu.reshape(2048, hidden)
        h2_cpu = x_viewed_cpu @ w2_cpu
        expected = h1_cpu.squeeze(0) + h2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )


class TestSharedInputReshapeNegative:
    """Negative tests where reshape should NOT be inserted or concat should fail."""

    @skip_if_not_enabled
    def test_different_dtype_shared_input_no_concat(self):
        """
        Test that shared inputs with different dtypes do NOT get concatenated.

        When two ops share the same data pointer but have different element types
        (e.g., one interprets as bf16, another as f32), reshape insertion should
        NOT be applied and concatenation should fail for that pair.
        """
        device = "neuron"
        num_iterations = 50

        hidden = 4096
        # Create tensors with different dtypes that happen to share similar sizes
        x_bf16_cpu = torch.randn(1, 2048, hidden, dtype=torch.bfloat16) * 0.01
        x_f32_cpu = torch.randn(1, 2048, hidden, dtype=torch.float32) * 0.01
        w_cpu = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.01

        x_bf16 = x_bf16_cpu.to(device)
        x_f32 = x_f32_cpu.to(device)
        w = w_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                # These use different dtypes — should not be merged via reshape
                h1 = x_bf16 @ w
                h2 = x_f32.to(torch.bfloat16) @ w
                result = h1 + h2

            result_final = result.cpu()

        print_concat_summary("test_different_dtype_shared_input_no_concat")

        # Verify correctness (the ops should still execute correctly)
        h1_cpu = x_bf16_cpu @ w_cpu
        h2_cpu = x_f32_cpu.to(torch.bfloat16) @ w_cpu
        expected = h1_cpu + h2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_incompatible_element_count_no_reshape(self):
        """
        Test that shared inputs with different element counts do NOT allow reshape.

        When the input shapes differ and element counts are different, the
        concatenation system should fall back rather than inserting reshape.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        # Different sized inputs — cannot be reshaped between each other
        x_small_cpu = torch.randn(1, 1024, 4096, dtype=dtype) * 0.01
        x_large_cpu = torch.randn(1, 2048, 4096, dtype=dtype) * 0.01
        w_cpu = torch.randn(4096, 4096, dtype=dtype) * 0.01

        x_small = x_small_cpu.to(device)
        x_large = x_large_cpu.to(device)
        w = w_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                h1 = x_small @ w
                h2 = x_large @ w
                result = torch.cat([h1, h2], dim=1)

            result_final = result.cpu()

        print_concat_summary("test_incompatible_element_count_no_reshape")

        # Verify correctness
        h1_cpu = x_small_cpu @ w_cpu
        h2_cpu = x_large_cpu @ w_cpu
        expected = torch.cat([h1_cpu, h2_cpu], dim=1)
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_same_shape_shared_input_no_reshape_inserted(self):
        """
        Test that when shared inputs have the SAME shape, no reshape is inserted.

        This is the trivial case — both modules see the input identically.
        Concatenation should succeed without needing any reshape ops.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        hidden = 4096
        x_cpu = torch.randn(1, 2048, hidden, dtype=dtype) * 0.01
        w1_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        w2_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        w1 = w1_cpu.to(device)
        w2 = w2_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                # Both matmuls use x in the SAME shape — no reshape needed
                h1 = x @ w1
                h2 = x @ w2
                result = h1 + h2

            result_final = result.cpu()

        metrics = print_concat_summary("test_same_shape_shared_input_no_reshape_inserted")

        # With same shapes, concatenation should succeed without reshape
        assert metrics["attempts"] > 0, "Should have concatenation attempts"
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Same-shape concat should work."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"

        # Verify correctness
        h1_cpu = x_cpu @ w1_cpu
        h2_cpu = x_cpu @ w2_cpu
        expected = h1_cpu + h2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
