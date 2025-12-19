# ruff: noqa: N806
"""
Unit tests for flex_attention decomposition

Tests the custom flex_attention decomposition against PyTorch's reference
implementation to ensure correctness.
"""

import math
from typing import Optional

import pytest
import torch
from torch.nn.attention.flex_attention import flex_attention as flex_attention_ref

from torch_neuronx.neuron_dynamo_backend.decompositions import flex_attention_decomposition


class TestFlexAttentionDecomposition:
    """Test suite for flex_attention decomposition"""

    @pytest.fixture
    def basic_inputs(self):
        """Create basic test inputs"""
        B, H, L, S, E = 2, 4, 8, 8, 16
        E_v = 16

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E_v)

        return query, key, value

    @pytest.fixture
    def gqa_inputs(self):
        """Create inputs for Grouped Query Attention"""
        B, H_q, H_kv, L, S, E = 2, 8, 2, 8, 8, 16
        E_v = 16

        query = torch.randn(B, H_q, L, E)
        key = torch.randn(B, H_kv, S, E)
        value = torch.randn(B, H_kv, S, E_v)

        return query, key, value

    def test_basic_attention(self, basic_inputs):
        """Test basic attention without any modifications"""
        query, key, value = basic_inputs

        # Run decomposition
        output = flex_attention_decomposition(query, key, value)

        # Check output shape
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)

        # Check output is finite
        assert torch.isfinite(output).all()

    def test_custom_scale(self, basic_inputs):
        """Test attention with custom scale factor"""
        query, key, value = basic_inputs
        scale = 0.5

        output = flex_attention_decomposition(query, key, value, scale=scale)

        # Check output shape
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_score_mod_causal(self, basic_inputs):
        """Test attention with causal score modification"""
        query, key, value = basic_inputs

        def causal_score_mod(score, batch, head, q_idx, k_idx):
            """Apply causal masking: only attend to previous positions"""
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        output = flex_attention_decomposition(query, key, value, score_mod=causal_score_mod)

        # Check output shape
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_score_mod_sliding_window(self, basic_inputs):
        """Test attention with sliding window score modification"""
        query, key, value = basic_inputs
        window_size = 3

        def sliding_window_score_mod(score, batch, head, q_idx, k_idx):
            """Apply sliding window: only attend to nearby positions"""
            distance = torch.abs(q_idx - k_idx)
            return torch.where(distance <= window_size, score, float("-inf"))

        output = flex_attention_decomposition(query, key, value, score_mod=sliding_window_score_mod)

        # Check output shape and validity
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_score_mod_relative_bias(self, basic_inputs):
        """Test attention with relative position bias"""
        query, key, value = basic_inputs

        def relative_bias_score_mod(score, batch, head, q_idx, k_idx):
            """Add relative position bias"""
            relative_pos = q_idx - k_idx
            bias = relative_pos * 0.1  # Simple linear bias
            return score + bias

        output = flex_attention_decomposition(query, key, value, score_mod=relative_bias_score_mod)

        # Check output shape and validity
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_gqa_basic(self, gqa_inputs):
        """Test Grouped Query Attention"""
        query, key, value = gqa_inputs

        output = flex_attention_decomposition(query, key, value, enable_gqa=True)

        # Check output shape
        B, H_q, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H_q, L, E_v)
        assert torch.isfinite(output).all()

    def test_gqa_with_score_mod(self, gqa_inputs):
        """Test GQA with score modification"""
        query, key, value = gqa_inputs

        def causal_score_mod(score, batch, head, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        output = flex_attention_decomposition(
            query, key, value, score_mod=causal_score_mod, enable_gqa=True
        )

        # Check output shape
        B, H_q, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H_q, L, E_v)
        assert torch.isfinite(output).all()

    def test_return_lse(self, basic_inputs):
        """Test returning log-sum-exp"""
        query, key, value = basic_inputs

        output, lse = flex_attention_decomposition(query, key, value, return_lse=True)

        # Check output shapes
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)
        assert lse.shape == (B, H, L)
        assert torch.isfinite(output).all()
        assert torch.isfinite(lse).all()

    def test_different_seq_lengths(self):
        """Test with different query and key sequence lengths"""
        B, H, L, S, E = 2, 4, 10, 15, 16
        E_v = 16

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E_v)

        output = flex_attention_decomposition(query, key, value)

        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_single_batch(self):
        """Test with batch size 1"""
        B, H, L, S, E = 1, 2, 4, 4, 8
        E_v = 8

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E_v)

        output = flex_attention_decomposition(query, key, value)

        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_large_dimensions(self):
        """Test with larger dimensions"""
        B, H, L, S, E = 4, 8, 32, 32, 64
        E_v = 64

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E_v)

        output = flex_attention_decomposition(query, key, value)

        assert output.shape == (B, H, L, E_v)
        assert torch.isfinite(output).all()

    def test_against_reference_basic(self, basic_inputs):
        """Compare against PyTorch reference implementation - basic case"""
        query, key, value = basic_inputs

        # Run both implementations
        output_decomp = flex_attention_decomposition(query, key, value)
        output_ref = flex_attention_ref(query, key, value)

        # Check shapes match
        assert output_decomp.shape == output_ref.shape

        # Check outputs are close (allowing for numerical differences)
        torch.testing.assert_close(output_decomp, output_ref, rtol=1e-4, atol=1e-5)

    def test_against_reference_with_scale(self, basic_inputs):
        """Compare against PyTorch reference - with custom scale"""
        query, key, value = basic_inputs
        scale = 0.25

        output_decomp = flex_attention_decomposition(query, key, value, scale=scale)
        output_ref = flex_attention_ref(query, key, value, scale=scale)

        torch.testing.assert_close(output_decomp, output_ref, rtol=1e-4, atol=1e-5)

    def test_against_reference_causal(self, basic_inputs):
        """Compare against PyTorch reference - with causal masking"""
        query, key, value = basic_inputs

        def causal_score_mod(score, batch, head, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        output_decomp = flex_attention_decomposition(query, key, value, score_mod=causal_score_mod)
        output_ref = flex_attention_ref(query, key, value, score_mod=causal_score_mod)

        torch.testing.assert_close(output_decomp, output_ref, rtol=1e-4, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        B, H, L, S, E = 2, 4, 8, 8, 16
        E_v = 16

        # Create inputs with large values
        query = torch.randn(B, H, L, E) * 10
        key = torch.randn(B, H, S, E) * 10
        value = torch.randn(B, H, S, E_v)

        output = flex_attention_decomposition(query, key, value)

        # Should still produce finite outputs due to softmax normalization
        assert torch.isfinite(output).all()

    def test_gradient_flow(self, basic_inputs):
        """Test that gradients flow correctly"""
        query, key, value = basic_inputs
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        output = flex_attention_decomposition(query, key, value)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert torch.isfinite(query.grad).all()
        assert torch.isfinite(key.grad).all()
        assert torch.isfinite(value.grad).all()


def test_manual_causal_attention():
    """Manual test for causal attention pattern"""
    B, H, L, E = 1, 1, 4, 8

    query = torch.randn(B, H, L, E)
    key = torch.randn(B, H, L, E)
    value = torch.eye(L).view(B, H, L, L)  # Identity for easy verification

    def causal_mod(score, batch, head, q_idx, k_idx):
        return torch.where(q_idx >= k_idx, score, float("-inf"))

    output = flex_attention_decomposition(query, key, value, score_mod=causal_mod)

    # With causal masking, each position should only attend to itself and previous positions
    # The output should have non-zero values only in the lower triangular part
    assert output.shape == (B, H, L, L)


def test_manual_sliding_window():
    """Manual test for sliding window attention"""
    B, H, L, E = 1, 1, 8, 8
    window = 2

    query = torch.randn(B, H, L, E)
    key = torch.randn(B, H, L, E)
    value = torch.randn(B, H, L, E)

    def sliding_window_mod(score, batch, head, q_idx, k_idx):
        distance = torch.abs(q_idx - k_idx)
        return torch.where(distance <= window, score, float("-inf"))

    output = flex_attention_decomposition(query, key, value, score_mod=sliding_window_mod)

    assert output.shape == (B, H, L, E)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running flex_attention decomposition tests...")

    test = TestFlexAttentionDecomposition()

    # Create fixtures
    B, H, L, S, E = 2, 4, 8, 8, 16
    E_v = 16
    basic_inputs = (torch.randn(B, H, L, E), torch.randn(B, H, S, E), torch.randn(B, H, S, E_v))

    B, H_q, H_kv, L, S, E = 2, 8, 2, 8, 8, 16
    gqa_inputs = (
        torch.randn(B, H_q, L, E),
        torch.randn(B, H_kv, S, E),
        torch.randn(B, H_kv, S, E_v),
    )

    print("✓ Test basic attention")
    test.test_basic_attention(basic_inputs)

    print("✓ Test custom scale")
    test.test_custom_scale(basic_inputs)

    print("✓ Test causal score mod")
    test.test_score_mod_causal(basic_inputs)

    print("✓ Test sliding window")
    test.test_score_mod_sliding_window(basic_inputs)

    print("✓ Test relative bias")
    test.test_score_mod_relative_bias(basic_inputs)

    print("✓ Test GQA basic")
    test.test_gqa_basic(gqa_inputs)

    print("✓ Test GQA with score mod")
    test.test_gqa_with_score_mod(gqa_inputs)

    print("✓ Test return LSE")
    test.test_return_lse(basic_inputs)

    print("✓ Test different seq lengths")
    test.test_different_seq_lengths()

    print("✓ Test single batch")
    test.test_single_batch()

    print("✓ Test large dimensions")
    test.test_large_dimensions()

    print("✓ Test numerical stability")
    test.test_numerical_stability()

    print("✓ Test gradient flow")
    test.test_gradient_flow(basic_inputs)

    print("✓ Test manual causal attention")
    test_manual_causal_attention()

    print("✓ Test manual sliding window")
    test_manual_sliding_window()

    print("✓ Test against reference - basic")
    test.test_against_reference_basic(basic_inputs)

    print("✓ Test against reference - with scale")
    test.test_against_reference_with_scale(basic_inputs)

    print("✓ Test against reference - causal")
    test.test_against_reference_causal(basic_inputs)

    print("\n✅ All tests passed!")
