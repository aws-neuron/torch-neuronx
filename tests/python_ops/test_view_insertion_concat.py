"""
Tests for view/reshape insertion in operation concatenation.

Verifies that operations following a view/reshape/unsqueeze (metadata-only ops
not dispatched to Neuron) can still be concatenated with their producer.

The key scenario: _fused_rms_norm produces output with shape [B, S, H, D],
but the downstream op sees it as [B, S, H, 1, D] due to an unsqueeze view.
The concatenation system must insert a stablehlo.reshape to bridge this.

Uses metrics for precise assertions. Optionally enable double buffering
and debug mode for additional coverage.

Requires:
    TORCH_NEURONX_ENABLE_CONCATENATION=1

Optional:
    TORCH_NEURONX_ENABLE_DOUBLE_BUFFERING=1
    TORCH_NEURONX_CONCATENATION_DEBUG_MODE=1 (analysis only — not for correctness tests)

Run with:
    TORCH_NEURONX_ENABLE_CONCATENATION=1 \
    TORCH_NEURONX_ENABLE_DOUBLE_BUFFERING=1 \
    pytest tests/python_ops/test_view_insertion_concat.py -xvs
"""

import os

import pytest
import torch

import torch_neuronx
import torch_neuronx.metrics as met
from tests.utils.neuron_test_utils import (
    get_executed_op_list,
    track_neuron_ops,
)


def is_enabled():
    return os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"


def is_debug_mode():
    return os.environ.get("TORCH_NEURONX_CONCATENATION_DEBUG_MODE", "0") == "1"


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
    """Print concatenation and double buffer metrics summary."""
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

    # Double buffer metrics if available
    db_pruned = met.get_counter_value("DoubleBuffer.OutputsPruned")
    if db_pruned is not None:
        db_entries = met.get_counter_value("DoubleBuffer.EntriesCreated") or 0
        db_consumed = met.get_counter_value("DoubleBuffer.CandidatesConsumedByInput") or 0
        print(f"  DB entries: {db_entries}")
        print(f"  DB outputs pruned: {db_pruned}")
        print(f"  DB candidates consumed: {db_consumed}")

    # Concatenated groups
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


class TestViewInsertionConcat:
    """Tests verifying that view/reshape ops don't block concatenation."""

    @skip_if_not_enabled
    def test_rms_norm_with_matmul_concatenates(self):
        """
        Test that rms_norm + matmul pattern triggers concatenation.

        Pattern: matmul -> rms_norm -> [view] -> matmul
        The matmuls are fusible boundaries. rms_norm sits between them and
        should be concatenated with adjacent ops despite view-induced shape
        mismatches.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        # Large tensors for compute backpressure
        hidden = 4096
        x_cpu = torch.randn(1, 2048, hidden, dtype=dtype) * 0.01
        w1_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        w2_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        norm_w_cpu = torch.randn(hidden, dtype=dtype)

        x = x_cpu.to(device)
        w1 = w1_cpu.to(device)
        w2 = w2_cpu.to(device)
        norm_w = norm_w_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                h = x @ w1  # matmul (fusible boundary)
                normed = torch.rms_norm(h, [hidden], norm_w)  # rms_norm (may have view)
                result = normed @ w2  # matmul (next fusible boundary)

            result_final = result.cpu()

        metrics = print_concat_summary("test_rms_norm_with_matmul_concatenates")

        # Assert concatenation was attempted and succeeded
        assert metrics["attempts"] > 0, (
            "No concatenation attempts. The matmul boundaries should trigger "
            "accumulation and concatenation."
        )
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Reshape insertion may be broken."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"
        assert (
            metrics["ops_concat"] >= 2
        ), f"Expected at least 2 ops concatenated, got {metrics['ops_concat']}"

        # Verify that concatenated groups contain the expected ops
        concat_groups = metrics["concat_groups"]
        assert len(concat_groups) > 0, "Should have recorded concatenated op groups"
        all_ops = []
        for ops in concat_groups.values():
            all_ops.extend(ops)
        assert any(
            "rms_norm" in op or "fused_rms_norm" in op for op in all_ops
        ), f"Expected rms_norm in concatenated ops, got: {all_ops}"

        # Verify correctness
        h_cpu = x_cpu @ w1_cpu
        normed_cpu = torch.rms_norm(h_cpu, [hidden], norm_w_cpu)
        expected = normed_cpu @ w2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_matmul_view_mul_concatenates(self):
        """
        Test that matmul -> view -> mul between matmul boundaries concatenates.

        Pattern: matmul -> [unsqueeze view] -> mul -> [squeeze view] -> matmul
        The unsqueeze/squeeze are metadata-only views. The mul op between
        matmul boundaries should be concatenated despite the shape mismatch
        caused by the view.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        batch = 8
        seq = 256
        hidden = 4096
        x_cpu = torch.randn(batch, seq, hidden, dtype=dtype) * 0.01
        w1_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        w2_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        scale_cpu = torch.randn(batch, seq, 1, hidden, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        w1 = w1_cpu.to(device)
        w2 = w2_cpu.to(device)
        scale = scale_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                h = x @ w1  # matmul boundary: [B, S, H]
                h_4d = h.unsqueeze(2)  # view: [B, S, 1, H] — metadata only
                h_scaled = h_4d * scale  # mul with shape [B, S, 1, H]
                h_3d = h_scaled.squeeze(2)  # view: [B, S, H] — metadata only
                result = h_3d @ w2  # matmul boundary

            result_final = result.cpu()

        metrics = print_concat_summary("test_matmul_view_mul_concatenates")

        # Concatenation should be attempted and succeed between the matmul boundaries
        assert metrics["attempts"] > 0, "Should have concatenation attempts"
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Reshape insertion may be broken."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"

        # The mul op should appear in a concatenated group (between matmul boundaries)
        concat_groups = metrics["concat_groups"]
        assert len(concat_groups) > 0, "Should have recorded concatenated op groups"
        all_ops = []
        for ops in concat_groups.values():
            all_ops.extend(ops)
        assert any(
            "mul" in op for op in all_ops
        ), f"Expected mul in concatenated ops, got: {all_ops}"

        # Verify correctness
        h_cpu = x_cpu @ w1_cpu
        h_4d_cpu = h_cpu.unsqueeze(2)
        h_scaled_cpu = h_4d_cpu * scale_cpu
        h_3d_cpu = h_scaled_cpu.squeeze(2)
        expected = h_3d_cpu @ w2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_unsqueeze_between_ops_concatenates(self):
        """
        Test that unsqueeze (a common view op) between compute ops doesn't break concat.

        Pattern: matmul -> unsqueeze(dim) -> mul
        unsqueeze adds a size-1 dimension (metadata-only, same data pointer).
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        x_cpu = torch.randn(1, 2048, 4096, dtype=dtype) * 0.01
        w_cpu = torch.randn(4096, 4096, dtype=dtype) * 0.01
        scale_cpu = torch.randn(1, 2048, 1, 4096, dtype=dtype) * 0.01

        x = x_cpu.to(device)
        w = w_cpu.to(device)
        scale = scale_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                h = x @ w
                h_expanded = h.unsqueeze(2)
                result = h_expanded * scale

            result_final = result.cpu()

        metrics = print_concat_summary("test_unsqueeze_between_ops_concatenates")

        assert metrics["attempts"] > 0, "Should have concatenation attempts"
        assert metrics["successes"] > 0, (
            f"No concatenation successes (attempts={metrics['attempts']}, "
            f"fallbacks={metrics['fallbacks']}). Reshape insertion may be broken."
        )
        assert metrics["groups"] > 0, "Should have created concatenated groups"

        # The mul op should be concatenated (between matmul boundaries, past unsqueeze view)
        concat_groups = metrics["concat_groups"]
        assert len(concat_groups) > 0, "Should have recorded concatenated op groups"
        all_ops = []
        for ops in concat_groups.values():
            all_ops.extend(ops)
        assert any(
            "mul" in op for op in all_ops
        ), f"Expected mul in concatenated ops, got: {all_ops}"

        # Verify correctness
        h_cpu = x_cpu @ w_cpu
        h_exp_cpu = h_cpu.unsqueeze(2)
        expected = h_exp_cpu * scale_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_concatenation_fallbacks_reduced(self):
        """
        Verify that the view insertion fix reduces fallback count.

        Pattern: matmul -> rms_norm -> matmul -> relu (between matmul boundaries)
        Without view insertion: rms_norm groups fail to merge (type mismatch).
        With view insertion: they should succeed.
        """
        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        hidden = 4096
        x_cpu = torch.randn(1, 2048, hidden, dtype=dtype) * 0.01
        w1_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        w2_cpu = torch.randn(hidden, hidden, dtype=dtype) * 0.01
        norm_w_cpu = torch.randn(hidden, dtype=dtype)

        x = x_cpu.to(device)
        w1 = w1_cpu.to(device)
        w2 = w2_cpu.to(device)
        norm_w = norm_w_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                h = x @ w1
                normed = torch.rms_norm(h, [hidden], norm_w)
                result = normed @ w2

            result_final = result.cpu()

        metrics = print_concat_summary("test_concatenation_fallbacks_reduced")

        successes = metrics["successes"]
        fallbacks = metrics["fallbacks"]

        # With view insertion, most concat attempts should succeed
        if successes + fallbacks > 0:
            fallback_rate = fallbacks / (successes + fallbacks)
            print(f"  Fallback rate: {fallback_rate:.1%}")

        # Verify correctness
        h_cpu = x_cpu @ w1_cpu
        normed_cpu = torch.rms_norm(h_cpu, [hidden], norm_w_cpu)
        expected = normed_cpu @ w2_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )

    @skip_if_not_enabled
    def test_double_buffer_pruning_with_views(self):
        """
        Test that double buffering prunes outputs correctly even when
        view insertion is active. Combines both features.

        Requires TORCH_NEURONX_ENABLE_DOUBLE_BUFFERING=1.
        """
        if os.environ.get("TORCH_NEURONX_ENABLE_DOUBLE_BUFFERING", "0") != "1":
            pytest.skip("Requires TORCH_NEURONX_ENABLE_DOUBLE_BUFFERING=1")

        device = "neuron"
        dtype = torch.bfloat16
        num_iterations = 50

        x_cpu = torch.randn(1, 2048, 4096, dtype=dtype) * 0.01
        w_cpu = torch.randn(4096, 4096, dtype=dtype) * 0.01
        norm_w_cpu = torch.randn(4096, dtype=dtype)

        x = x_cpu.to(device)
        w = w_cpu.to(device)
        norm_w = norm_w_cpu.to(device)

        with track_neuron_ops():
            for _ in range(num_iterations):
                normed = torch.rms_norm(x, [4096], norm_w)
                h = normed @ w
                result = torch.relu(h) + x

            result_final = result.cpu()

        print_concat_summary("test_double_buffer_pruning_with_views")

        # Check double buffer pruning metrics
        db_pruned = met.get_counter_value("DoubleBuffer.OutputsPruned") or 0
        db_entries = met.get_counter_value("DoubleBuffer.EntriesCreated") or 0

        print(f"\n  Double buffer entries: {db_entries}")
        print(f"  Double buffer outputs pruned: {db_pruned}")

        assert db_entries > 0, "Double buffer should have created entries"

        # Verify correctness
        normed_cpu = torch.rms_norm(x_cpu, [4096], norm_w_cpu)
        expected = torch.relu(normed_cpu @ w_cpu) + x_cpu
        assert torch.allclose(result_final, expected, rtol=1e-2, atol=1e-2), (
            f"Result mismatch! max_diff=" f"{(result_final - expected).abs().max().item():.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
