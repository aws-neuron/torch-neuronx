"""
Tests for operation concatenation optimization correctness.

Verifies that the concatenation engine correctly batches operations
and produces the same results as CPU execution. The concatenation
system optimizes matmul-heavy workloads by accumulating operations
and flushing them in batches for better hardware utilization.

NOTE: These tests require the following environment variables:
    os.environ["TORCH_NEURONX_SKIP_INTERMEDIATES"] = "1"
    os.environ["TORCH_NEURONX_ENABLE_CONCATENATION"] = "1"
These must be set BEFORE running the tests to ensure concatenation
optimizations work correctly and the tests validate the intended behavior.

IMPORTANT: To ensure concatenation happens:
1. Avoid calling .cpu() inside loops - only call ONCE at the end
2. Avoid torch.tensor() with scalars inside loops - creates host copies
3. Create tensors on CPU first, copy to device outside the loop

Run with: pytest tests/python_ops/test_concatenation_ops.py -v
"""

import os

import pytest
import torch

import torch_neuronx
import torch_neuronx.metrics as met
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    get_executed_op_list,
    track_neuron_ops,
)


def is_concatenation_enabled():
    """Check if concatenation is enabled."""
    return os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"


def is_skip_intermediates_enabled():
    """Check if skip intermediates is enabled."""
    return os.environ.get("TORCH_NEURONX_SKIP_INTERMEDIATES", "0") == "1"


def skip_if_concatenation_disabled():
    """Skip test if concatenation is disabled."""
    return pytest.mark.skipif(
        not is_concatenation_enabled() or not is_skip_intermediates_enabled(),
        reason=(
            "Concatenation tests require TORCH_NEURONX_ENABLE_CONCATENATION=1 "
            "and TORCH_NEURONX_SKIP_INTERMEDIATES=1"
        ),
    )


def assert_concatenation_occurred(test_name: str = ""):
    """
    Assert that concatenation occurred using telemetry metrics.

    Uses NeuronMetrics counters (Concatenation.GroupsCreated, Concatenation.Successes)
    as the primary signal — these are more deterministic than checking for pipe-separated
    op names since they're incremented at group creation time, not execution time.

    Falls back to checking executed op names for pipe-separated patterns as secondary signal.

    This should be called after torch.neuron.synchronize().

    Args:
        test_name: Optional test name for error messages
    """
    torch.neuron.synchronize()

    # Primary check: telemetry metrics (deterministic)
    groups_created = met.get_counter_value("Concatenation.GroupsCreated")
    successes = met.get_counter_value("Concatenation.Successes")
    attempts = met.get_counter_value("Concatenation.Attempts")

    # Secondary check: pipe-separated op names in executed ops
    op_list = get_executed_op_list()
    executed_ops = op_list["executed"]
    concatenated_ops = [op for op in executed_ops if "|" in op]

    # Also check ConcatenatedOpTracker
    concat_groups = torch_neuronx.get_concatenated_ops()

    # Use metrics as primary signal (most deterministic)
    metrics_show_concat = (groups_created is not None and groups_created > 0) or (
        successes is not None and successes > 0
    )
    ops_show_concat = len(concatenated_ops) > 0
    tracker_show_concat = len(concat_groups) > 0

    assert metrics_show_concat or ops_show_concat or tracker_show_concat, (
        f"{test_name}: Concatenation did not occur. "
        f"Metrics: Attempts={attempts}, GroupsCreated={groups_created}, Successes={successes}. "
        f"Concatenated ops in executed: {len(concatenated_ops)}. "
        f"ConcatenatedOpTracker groups: {len(concat_groups)}. "
        f"Executed ops: {executed_ops}"
    )

    return concatenated_ops


@pytest.fixture(autouse=True)
def enable_metrics():
    """Enable metrics collection and clear state before each test."""
    met.set_enabled(True)
    met.clear_all()
    torch_neuronx.clear_op_tracking()
    yield
    # Metrics persist after test for debugging; cleared on next test's setup


class TestConcatenationBasicOps:
    """Test suite for basic concatenation operation correctness."""

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_iterations", [50, 60, 80])
    def test_matmul_loop_precision(self, dtype, num_iterations):
        """
        Test matmul operations in a loop produce correct results.

        Pattern: Multiple independent matmul projections (Q/K/V-style) per
        iteration, mimicking a 70B-model transformer layer. This ensures:
        1. Large ops (8192-dim) create enough compute backpressure to exceed
           the queue_low_threshold, routing ops through concatenation.
        2. Multiple independent same-shape matmuls per iteration allow the
           concatenation engine to form groups.
        """
        device = "neuron"

        # 70B-model-like dimensions: (batch*seq, hidden) @ (hidden, hidden)
        # For bfloat16, scale inputs down to keep output magnitudes small enough
        # that hardware bf16 rounding (~1 ULP) stays within tight atol.
        # At scale=1.0, outputs reach ~700 and bf16 ULP=2.0 exceeds atol=0.5.
        # At scale=0.1, outputs reach ~7 and bf16 ULP=0.016 fits atol=0.05.
        input_scale = 0.1 if dtype == torch.bfloat16 else 1.0

        x_cpu = torch.randn(128, 8192, dtype=dtype) * input_scale
        w1_cpu = torch.randn(8192, 8192, dtype=dtype) * input_scale
        w2_cpu = torch.randn(8192, 8192, dtype=dtype) * input_scale
        w3_cpu = torch.randn(8192, 8192, dtype=dtype) * input_scale

        # Copy to Neuron ONCE, outside the tracking context to avoid copy ops
        x_neuron = x_cpu.to(device)
        w1_neuron = w1_cpu.to(device)
        w2_neuron = w2_cpu.to(device)
        w3_neuron = w3_cpu.to(device)

        with track_neuron_ops():
            # Run all iterations on Neuron WITHOUT any host copies
            # Three independent projections per iteration (like Q/K/V)
            for _i in range(num_iterations):
                r1 = x_neuron @ w1_neuron
                r2 = x_neuron @ w2_neuron
                r3 = x_neuron @ w3_neuron
                result = r1 + r2 + r3

            # Single .cpu() at the end - fusion boundary
            result_final = result.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred(f"test_matmul_loop_precision[{dtype},{num_iterations}]")

            # Verify results
            r1_cpu = x_cpu @ w1_cpu
            r2_cpu = x_cpu @ w2_cpu
            r3_cpu = x_cpu @ w3_cpu
            result_cpu = r1_cpu + r2_cpu + r3_cpu

            if dtype == torch.float32:
                assert torch.allclose(
                    result_final, result_cpu, rtol=1e-3, atol=1e-3
                ), "Matmul produced incorrect results"
            else:  # bfloat16 with scaled inputs — observed max diff ~0.016 at scale=0.1
                max_diff = (result_final.float() - result_cpu.float()).abs().max().item()
                assert torch.allclose(result_final, result_cpu, rtol=1e-2, atol=1e-2), (
                    f"Matmul produced incorrect results for bfloat16 " f"(max_diff={max_diff:.4f})"
                )

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_add_loop_precision(self, dtype):
        """
        Test matmul followed by add (bias) operations in a loop.

        Pattern: Three independent matmul projections (Q/K/V-style) per
        iteration using 70B-model dimensions, followed by a single bias
        addition. Keeps the same proven concat-friendly pattern as
        test_matmul_loop_precision while also exercising bias addition.
        """
        device = "neuron"
        num_iterations = 50

        # 70B-model-like dimensions with a different batch size (64) than
        # test_matmul_loop_precision (128) to ensure distinct cache keys and
        # force compilation, creating the queue backpressure needed for concat.
        input_scale = 0.1 if dtype == torch.bfloat16 else 1.0

        x_cpu = torch.randn(64, 8192, dtype=dtype) * input_scale
        w1_cpu = torch.randn(8192, 8192, dtype=dtype) * input_scale
        w2_cpu = torch.randn(8192, 8192, dtype=dtype) * input_scale
        w3_cpu = torch.randn(8192, 8192, dtype=dtype) * input_scale
        bias_cpu = torch.randn(64, 8192, dtype=dtype) * input_scale

        # Copy to Neuron ONCE
        x_neuron = x_cpu.to(device)
        w1_neuron = w1_cpu.to(device)
        w2_neuron = w2_cpu.to(device)
        w3_neuron = w3_cpu.to(device)
        bias_neuron = bias_cpu.to(device)

        with track_neuron_ops():
            # Three independent matmul projections, then sum with bias.
            # Same pattern as test_matmul_loop_precision + bias term.
            for _i in range(num_iterations):
                r1 = x_neuron @ w1_neuron
                r2 = x_neuron @ w2_neuron
                r3 = x_neuron @ w3_neuron
                result = r1 + r2 + r3 + bias_neuron

            # Single .cpu() at the end
            result_final = result.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred(f"test_matmul_add_loop_precision[{dtype}]")

            # Verify results
            r1_cpu = x_cpu @ w1_cpu
            r2_cpu = x_cpu @ w2_cpu
            r3_cpu = x_cpu @ w3_cpu
            result_cpu = r1_cpu + r2_cpu + r3_cpu + bias_cpu

            if dtype == torch.float32:
                assert torch.allclose(
                    result_final, result_cpu, rtol=1e-3, atol=1e-3
                ), "Matmul+add produced incorrect results"
            else:  # bfloat16 with scaled inputs
                max_diff = (result_final.float() - result_cpu.float()).abs().max().item()
                assert torch.allclose(result_final, result_cpu, rtol=1e-2, atol=1e-2), (
                    f"Matmul+add produced incorrect results for bfloat16 "
                    f"(max_diff={max_diff:.4f})"
                )


class TestConcatenationAccumulationMode:
    """Test suite for concatenation accumulation mode behavior."""

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_accumulation_with_multiple_ops(self, dtype):
        """
        Test that multiple operations get accumulated and concatenated.

        Tests the fix for duplicate input handling (x + x, c + c + c patterns).
        """
        device = "neuron"
        num_iterations = 20

        # Create all tensors on CPU first
        a_cpu = torch.randn(64, 128, dtype=dtype)
        b_cpu = torch.randn(128, 64, dtype=dtype)

        # Copy to Neuron ONCE
        a = a_cpu.to(device)
        b = b_cpu.to(device)

        with track_neuron_ops():
            for _i in range(num_iterations):
                c = a @ b
                d = c + c + c
                e = torch.relu(d)
                f = e + e + e

            # Single .cpu() - fusion boundary
            f_result = f.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred("test_accumulation_with_multiple_ops")

            # Verify results
            c_cpu = a_cpu @ b_cpu
            d_cpu = c_cpu + c_cpu + c_cpu
            e_cpu = torch.relu(d_cpu)
            f_cpu = e_cpu + e_cpu + e_cpu

            # Relaxed tolerance for many iterations
            assert torch.allclose(
                f_result, f_cpu, rtol=1e-2, atol=1e-2
            ), "Accumulated operations produced incorrect results"


class TestConcatenationIterativeWorkload:
    """Test suite for iterative ML workload patterns."""

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("num_iterations", [50, 80])
    def test_transformer_like_pattern(self, dtype, num_iterations):
        """
        Test transformer-like operation pattern in a loop.

        Note: Only testing float32 as bfloat16 has numerical precision issues
        with multi-iteration softmax patterns.
        """
        device = "neuron"
        batch_size = 8
        seq_len = 64
        d_model = 128
        d_head = 32
        scale_val = 1.0 / (d_head**0.5)

        # Create all tensors on CPU first
        w_q_cpu = torch.randn(d_model, d_head, dtype=dtype)
        w_k_cpu = torch.randn(d_model, d_head, dtype=dtype)
        w_v_cpu = torch.randn(d_model, d_head, dtype=dtype)
        w_o_cpu = torch.randn(d_head, d_model, dtype=dtype)
        x_cpu = torch.randn(batch_size, seq_len, d_model, dtype=dtype)

        # Copy to Neuron ONCE
        w_q = w_q_cpu.to(device)
        w_k = w_k_cpu.to(device)
        w_v = w_v_cpu.to(device)
        w_o = w_o_cpu.to(device)
        x = x_cpu.to(device)

        with track_neuron_ops():
            for _i in range(num_iterations):
                q = torch.matmul(x, w_q)
                k = torch.matmul(x, w_k)
                v = torch.matmul(x, w_v)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale_val
                attn_weights = torch.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, v)
                output = torch.matmul(attn_output, w_o)

            # Single .cpu() at the end
            output_final = output.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred(
                f"test_transformer_like_pattern[{dtype},{num_iterations}]"
            )

            # Verify results
            q_cpu = torch.matmul(x_cpu, w_q_cpu)
            k_cpu = torch.matmul(x_cpu, w_k_cpu)
            v_cpu = torch.matmul(x_cpu, w_v_cpu)
            scores_cpu = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale_val
            attn_weights_cpu = torch.softmax(scores_cpu, dim=-1)
            attn_output_cpu = torch.matmul(attn_weights_cpu, v_cpu)
            output_cpu = torch.matmul(attn_output_cpu, w_o_cpu)

            assert torch.allclose(
                output_final, output_cpu, rtol=1e-2, atol=1e-2
            ), "Transformer pattern produced incorrect results"

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_mlp_pattern_iterations(self, dtype):
        """
        Test MLP (feed-forward) pattern in iterations.
        """
        device = "neuron"
        num_iterations = 50
        batch_size = 16
        in_features = 128
        hidden_features = 256
        out_features = 128

        # Create all tensors on CPU first
        w1_cpu = torch.randn(in_features, hidden_features, dtype=dtype)
        b1_cpu = torch.randn(hidden_features, dtype=dtype)
        w2_cpu = torch.randn(hidden_features, out_features, dtype=dtype)
        b2_cpu = torch.randn(out_features, dtype=dtype)
        x_cpu = torch.randn(batch_size, in_features, dtype=dtype)

        # Copy to Neuron ONCE
        w1 = w1_cpu.to(device)
        b1 = b1_cpu.to(device)
        w2 = w2_cpu.to(device)
        b2 = b2_cpu.to(device)
        x = x_cpu.to(device)

        with track_neuron_ops():
            for _i in range(num_iterations):
                hidden = x @ w1 + b1
                hidden_act = torch.relu(hidden)
                output = hidden_act @ w2 + b2

            # Single .cpu() at the end
            output_final = output.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred("test_mlp_pattern_iterations")

            # Verify results
            hidden_cpu = x_cpu @ w1_cpu + b1_cpu
            hidden_act_cpu = torch.relu(hidden_cpu)
            output_cpu = hidden_act_cpu @ w2_cpu + b2_cpu

            assert torch.allclose(
                output_final, output_cpu, rtol=1e-4, atol=1e-4
            ), "MLP pattern produced incorrect results"


class TestConcatenationVerification:
    """Tests that verify concatenation is actually happening using telemetry metrics."""

    @skip_if_concatenation_disabled()
    def test_concatenation_metrics(self):
        """
        Verify concatenation using telemetry metrics (deterministic).

        Checks Concatenation.GroupsCreated, Concatenation.OpsConcatenated,
        and ConcatenatedOpTracker for structured group data.
        """
        device = "neuron"
        dtype = torch.float32
        num_iterations = 50

        # Create tensors on CPU first
        a_cpu = torch.randn(64, 128, dtype=dtype)
        b_cpu = torch.randn(128, 64, dtype=dtype)

        # Copy to Neuron ONCE
        a = a_cpu.to(device)
        b = b_cpu.to(device)

        with track_neuron_ops():
            for _i in range(num_iterations):
                c = a @ b
                d = c + c
                e = torch.relu(d)
                f = e + e

            # Single .cpu() at the end
            f.cpu()

            # Verify via telemetry metrics
            assert_concatenation_occurred("test_concatenation_metrics")

            # Check structured metrics
            concat_metrics = met.get_metrics_dict("Concatenation")
            groups_created = concat_metrics.get("Concatenation.GroupsCreated", 0)
            ops_concatenated = concat_metrics.get("Concatenation.OpsConcatenated", 0)

            print("✓ Concatenation metrics:")
            print(f"  Groups created: {groups_created}")
            print(f"  Ops concatenated: {ops_concatenated}")

            # Check ConcatenatedOpTracker for group details
            concat_groups = torch_neuronx.get_concatenated_ops()
            if concat_groups:
                print(f"  Tracked groups: {len(concat_groups)}")
                for name, ops in concat_groups.items():
                    print(f"    {name} <- {ops}")

            # Check execution counts
            exec_counts = torch_neuronx.get_concatenated_op_execution_counts()
            if exec_counts:
                print("  Execution counts:")
                for name, count in exec_counts.items():
                    cache_hits = max(0, count - 1)
                    print(f"    {name}: {count} executions ({cache_hits} cache hits)")

    @skip_if_concatenation_disabled()
    def test_concatenation_status_diagnostic(self):
        """
        Test the automated concatenation status diagnostic.
        """
        device = "neuron"
        dtype = torch.float32
        num_iterations = 50

        a_cpu = torch.randn(64, 128, dtype=dtype)
        b_cpu = torch.randn(128, 64, dtype=dtype)

        a = a_cpu.to(device)
        b = b_cpu.to(device)

        with track_neuron_ops():
            for _i in range(num_iterations):
                c = a @ b
                d = c + c
                e = torch.pow(d, 2)

            e.cpu()

            # Use the automated diagnostic
            status = torch_neuronx.get_concatenation_status()

            print(f"Concatenation status: {status['diagnosis']}")
            print(f"  Enabled: {status['enabled']}")
            print(f"  Active: {status['is_active']}")
            print(f"  Concatenated ops in executed: {len(status['concatenated_ops_in_executed'])}")
            print(f"  Groups tracked: {status['concatenated_groups_tracked']}")

            # The diagnostic should confirm concatenation is enabled
            assert status["enabled"], "Concatenation should be enabled"


class TestConcatenationNullTensorPtrRegression:
    """Regression tests verifying concatenated kernels execute correctly.

    Validates that IrNodeManager::CreateNodeFromOperationContext properly
    propagates TensorDataRef (including NeuronTensorPtr) so concatenated
    kernels have valid tensor backing for both lazy-alloc and non-lazy-alloc
    execution paths.
    """

    @skip_if_concatenation_disabled()
    def test_elementwise_chain_concatenation(self):
        """Element-wise ops between matmul boundaries get concatenated and must not crash."""
        device = "neuron"
        dtype = torch.float32

        a_cpu = torch.randn(64, 128, dtype=dtype)
        b_cpu = torch.randn(64, 128, dtype=dtype)

        a = a_cpu.to(device)
        b = b_cpu.to(device)

        # Element-wise ops that get buffered and concatenated
        c = a + b
        d = c * a
        e = d - b
        f = torch.relu(e)

        result = f.cpu()

        # Verify correctness against CPU
        expected = torch.relu((a_cpu + b_cpu) * a_cpu - b_cpu)
        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

    @skip_if_concatenation_disabled()
    def test_cross_entropy_loss_with_concatenation(self):
        """Cross-entropy loss dispatches ne/where/add/gather ops that get concatenated."""
        device = "neuron"

        logits = torch.randn(4, 1000, device=device, requires_grad=True)
        targets = torch.randint(0, 1000, (4,), device=device)

        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        loss_val = loss.detach().cpu()
        assert not torch.isnan(loss_val), "Loss should not be NaN"
        assert not torch.isinf(loss_val), "Loss should not be Inf"
        assert logits.grad is not None, "Gradients should flow back"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
