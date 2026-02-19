"""
Tests for operation concatenation optimization correctness.

Verifies that the concatenation engine correctly batches operations
and produces the same results as CPU execution. The concatenation
system optimizes matmul-heavy workloads by accumulating operations
and flushing them in batches for better hardware utilization.

NOTE: These tests require the following environment variables:
    os.environ["TORCH_NEURONX_ENABLE_STABLEHLO"] = "1"
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

from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    get_executed_op_list,
    track_neuron_ops,
)


def is_concatenation_enabled():
    """Check if concatenation is enabled."""
    return os.environ.get("TORCH_NEURONX_ENABLE_CONCATENATION", "0") == "1"


def is_stablehlo_enabled():
    """Check if StableHLO is enabled."""
    return os.environ.get("TORCH_NEURONX_ENABLE_STABLEHLO", "1") not in ("0", "false")


def skip_if_concatenation_disabled():
    """Skip test if concatenation is disabled."""
    return pytest.mark.skipif(
        not is_concatenation_enabled() or not is_stablehlo_enabled(),
        reason=(
            "Concatenation tests require TORCH_NEURONX_ENABLE_CONCATENATION=1 "
            "and TORCH_NEURONX_ENABLE_STABLEHLO=1"
        ),
    )


def assert_concatenation_occurred(test_name: str = ""):
    """
    Assert that concatenation occurred by checking for pipe-separated op names.

    Concatenated operations have '|' separator in their names, e.g.:
    - 'mul|pow|aten::mm'
    - 'add_default|pow|mean'
    - 'silu|aten::linear'

    This should be called after the .cpu() fusion boundary.

    Args:
        test_name: Optional test name for error messages
    """
    op_list = get_executed_op_list()
    executed_ops = op_list["executed"]

    # Find concatenated operations (ops with '|' in name)
    concatenated_ops = [op for op in executed_ops if "|" in op]

    assert len(concatenated_ops) > 0, (
        f"{test_name}: Expected concatenated operations (pipe-separated op names) "
        f"but found none. Executed ops: {executed_ops}"
    )

    return concatenated_ops


class TestConcatenationBasicOps:
    """Test suite for basic concatenation operation correctness."""

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("num_iterations", [50, 60, 80])
    def test_matmul_loop_precision(self, dtype, num_iterations):
        """
        Test matmul operations in a loop produce correct results.

        Pattern: Multiple matmul + add operations in a loop.
        """
        device = "neuron"

        # Create all tensors on CPU first
        a_cpu = torch.randn(128, 256, dtype=dtype)
        b_cpu = torch.randn(256, 512, dtype=dtype)

        # Copy to Neuron ONCE, outside the tracking context to avoid copy ops
        a_neuron = a_cpu.to(device)
        b_neuron = b_cpu.to(device)

        with track_neuron_ops():
            # Run all iterations on Neuron WITHOUT any host copies
            for _i in range(num_iterations):
                result = a_neuron @ b_neuron
                result = result + result

            # Single .cpu() at the end - fusion boundary
            result_final = result.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred(f"test_matmul_loop_precision[{dtype},{num_iterations}]")

            # Verify results
            result_cpu = a_cpu @ b_cpu
            result_cpu = result_cpu + result_cpu

            if dtype == torch.float32:
                assert torch.allclose(
                    result_final, result_cpu, rtol=1e-4, atol=1e-4
                ), "Matmul produced incorrect results"
            else:  # bfloat16
                assert torch.allclose(
                    result_final, result_cpu, rtol=1e-2, atol=1e-2
                ), "Matmul produced incorrect results for bfloat16"

    @skip_if_concatenation_disabled()
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matmul_add_loop_precision(self, dtype):
        """
        Test matmul followed by add operations in a loop.
        """
        device = "neuron"
        num_iterations = 50

        # Create all tensors on CPU first
        a_cpu = torch.randn(64, 128, dtype=dtype)
        b_cpu = torch.randn(128, 256, dtype=dtype)
        bias_cpu = torch.randn(64, 256, dtype=dtype)

        # Copy to Neuron ONCE
        a_neuron = a_cpu.to(device)
        b_neuron = b_cpu.to(device)
        bias_neuron = bias_cpu.to(device)

        with track_neuron_ops():
            for _i in range(num_iterations):
                result = a_neuron @ b_neuron
                result = result + bias_neuron

            # Single .cpu() at the end
            result_final = result.cpu()

            # Verify concatenation occurred
            assert_concatenation_occurred(f"test_matmul_add_loop_precision[{dtype}]")

            # Verify results
            result_cpu = (a_cpu @ b_cpu) + bias_cpu

            if dtype == torch.float32:
                assert torch.allclose(
                    result_final, result_cpu, rtol=1e-4, atol=1e-4
                ), "Matmul+add produced incorrect results"
            else:
                assert torch.allclose(
                    result_final, result_cpu, rtol=1e-2, atol=1e-2
                ), "Matmul+add produced incorrect results for bfloat16"


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
    """Tests that verify concatenation is actually happening."""

    @skip_if_concatenation_disabled()
    def test_concatenation_op_executed(self):
        """
        Verify that concatenated operations are executed.

        Concatenated ops appear as pipe-separated names like:
        - 'add_default|pow|mean'
        - 'mul|aten::linear'
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

            # Verify concatenation occurred
            concatenated_ops = assert_concatenation_occurred("test_concatenation_op_executed")

            print(f"✓ Concatenation working: found {len(concatenated_ops)} concatenated ops")
            for op in concatenated_ops:
                print(f"  - {op}")

    @skip_if_concatenation_disabled()
    def test_verify_concatenated_ops_in_executed_list(self):
        """
        Explicitly verify that concatenated operations (pipe-separated names)
        appear in the executed ops list.
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
                e = torch.pow(d, 2)

            # Single .cpu() at the end
            e.cpu()

            # Get executed operations
            op_list = get_executed_op_list()
            executed_ops = op_list["executed"]

            # Verify concatenation occurred
            concatenated_ops = assert_concatenation_occurred(
                "test_verify_concatenated_ops_in_executed_list"
            )

            # Report statistics
            num_ops = len(executed_ops)
            num_concatenated = len(concatenated_ops)
            concat_percentage = (num_concatenated / num_ops * 100) if num_ops > 0 else 0

            print(f"Total executed ops: {num_ops}")
            print(f"Concatenated ops: {num_concatenated} ({concat_percentage:.1f}%)")
            print(f"All executed: {executed_ops}")
            print(f"✓ Found {num_concatenated} concatenated operations:")
            for op in concatenated_ops:
                print(f"  - {op}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
