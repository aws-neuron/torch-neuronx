# ruff: noqa: N806, N812
"""
Unit tests comparing FlexAttention decomposition against SDPA

This test suite compares the custom flex_attention decomposition against
PyTorch's standard scaled_dot_product_attention to ensure correctness
and measure performance differences.
"""

import math
from collections.abc import Callable
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from torch_neuronx.neuron_dynamo_backend.decompositions import flex_attention_decomposition


class TestFlexAttentionVsSDPA:
    """Test suite comparing FlexAttention decomposition against SDPA"""

    @pytest.fixture(autouse=True)
    def set_seed(self):
        """Set manual seed for reproducibility"""
        torch.manual_seed(42)

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

    @pytest.fixture
    def long_sequence_inputs(self):
        """Create inputs with longer sequences"""
        B, H, L, S, E = 1, 4, 128, 128, 64
        E_v = 64

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E_v)

        return query, key, value

    def test_basic_attention_vs_sdpa(self, basic_inputs):
        """Test basic attention matches SDPA"""
        query, key, value = basic_inputs

        # FlexAttention decomposition (no score_mod = standard attention)
        flex_output = flex_attention_decomposition(query, key, value)

        # SDPA reference
        sdpa_output = F.scaled_dot_product_attention(query, key, value)

        # Check shapes match
        assert flex_output.shape == sdpa_output.shape

        # Check outputs are close
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_causal_attention_vs_sdpa(self, basic_inputs):
        """Test causal attention matches SDPA with is_causal=True"""
        query, key, value = basic_inputs

        # FlexAttention with causal score_mod
        def causal_score_mod(score, batch, head, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        flex_output = flex_attention_decomposition(query, key, value, score_mod=causal_score_mod)

        # SDPA with causal mask
        sdpa_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        # Check shapes match
        assert flex_output.shape == sdpa_output.shape

        # Check outputs are close
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_custom_scale_vs_sdpa(self, basic_inputs):
        """Test custom scale factor matches SDPA"""
        query, key, value = basic_inputs
        scale = 0.25

        # FlexAttention with custom scale
        flex_output = flex_attention_decomposition(query, key, value, scale=scale)

        # SDPA with custom scale
        sdpa_output = F.scaled_dot_product_attention(query, key, value, scale=scale)

        # Check outputs are close
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_sliding_window_attention(self, basic_inputs):
        """Test sliding window attention (not available in SDPA)"""
        query, key, value = basic_inputs
        window_size = 3

        def sliding_window_score_mod(score, batch, head, q_idx, k_idx):
            """Causal + sliding window"""
            causal_mask = q_idx >= k_idx
            window_mask = (q_idx - k_idx) <= window_size
            combined_mask = causal_mask & window_mask
            return torch.where(combined_mask, score, float("-inf"))

        # FlexAttention with sliding window
        flex_output = flex_attention_decomposition(
            query, key, value, score_mod=sliding_window_score_mod
        )

        # Check output shape and validity
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert flex_output.shape == (B, H, L, E_v)
        assert torch.isfinite(flex_output).all()

        # Verify sliding window behavior by checking attention pattern
        # For positions beyond window, output should be different from full causal
        def causal_only_score_mod(score, batch, head, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        causal_output = flex_attention_decomposition(
            query, key, value, score_mod=causal_only_score_mod
        )

        # Outputs should differ (sliding window is more restrictive)
        assert not torch.allclose(flex_output, causal_output, rtol=1e-4, atol=1e-5)

    def test_gqa_basic(self, gqa_inputs):
        """Test Grouped Query Attention"""
        query, key, value = gqa_inputs
        B, H_q, L, E = query.shape
        _, H_kv, S, E_v = value.shape

        # FlexAttention with GQA
        flex_output = flex_attention_decomposition(query, key, value, enable_gqa=True)

        # Manual GQA with SDPA (repeat key/value heads)
        repeat_factor = H_q // H_kv
        key_repeated = key.repeat_interleave(repeat_factor, dim=1)
        value_repeated = value.repeat_interleave(repeat_factor, dim=1)
        sdpa_output = F.scaled_dot_product_attention(query, key_repeated, value_repeated)

        # Check outputs match
        assert flex_output.shape == sdpa_output.shape
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_gqa_with_causal(self, gqa_inputs):
        """Test GQA with causal masking"""
        query, key, value = gqa_inputs
        B, H_q, L, E = query.shape
        _, H_kv, S, E_v = value.shape

        def causal_score_mod(score, batch, head, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        # FlexAttention with GQA + causal
        flex_output = flex_attention_decomposition(
            query, key, value, score_mod=causal_score_mod, enable_gqa=True
        )

        # Manual GQA with SDPA + causal
        repeat_factor = H_q // H_kv
        key_repeated = key.repeat_interleave(repeat_factor, dim=1)
        value_repeated = value.repeat_interleave(repeat_factor, dim=1)
        sdpa_output = F.scaled_dot_product_attention(
            query, key_repeated, value_repeated, is_causal=True
        )

        # Check outputs match
        assert flex_output.shape == sdpa_output.shape
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_different_seq_lengths(self):
        """Test with different query and key sequence lengths"""
        B, H, L, S, E = 2, 4, 10, 15, 16
        E_v = 16

        query = torch.randn(B, H, L, E)
        key = torch.randn(B, H, S, E)
        value = torch.randn(B, H, S, E_v)

        # FlexAttention
        flex_output = flex_attention_decomposition(query, key, value)

        # SDPA
        sdpa_output = F.scaled_dot_product_attention(query, key, value)

        # Check outputs match
        assert flex_output.shape == sdpa_output.shape
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_long_sequences(self, long_sequence_inputs):
        """Test with longer sequences (128 tokens)"""
        query, key, value = long_sequence_inputs

        # FlexAttention with causal
        def causal_score_mod(score, batch, head, q_idx, k_idx):
            return torch.where(q_idx >= k_idx, score, float("-inf"))

        flex_output = flex_attention_decomposition(query, key, value, score_mod=causal_score_mod)

        # SDPA with causal
        sdpa_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        # Check outputs match
        assert flex_output.shape == sdpa_output.shape
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)

    def test_gradient_flow_comparison(self, basic_inputs):
        """Test that gradients match between FlexAttention and SDPA"""
        query, key, value = basic_inputs

        # FlexAttention path
        query_flex = query.clone().requires_grad_(True)
        key_flex = key.clone().requires_grad_(True)
        value_flex = value.clone().requires_grad_(True)

        flex_output = flex_attention_decomposition(query_flex, key_flex, value_flex)
        flex_loss = flex_output.sum()
        flex_loss.backward()

        # SDPA path
        query_sdpa = query.clone().requires_grad_(True)
        key_sdpa = key.clone().requires_grad_(True)
        value_sdpa = value.clone().requires_grad_(True)

        sdpa_output = F.scaled_dot_product_attention(query_sdpa, key_sdpa, value_sdpa)
        sdpa_loss = sdpa_output.sum()
        sdpa_loss.backward()

        # Check gradients match
        torch.testing.assert_close(query_flex.grad, query_sdpa.grad, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(key_flex.grad, key_sdpa.grad, rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(value_flex.grad, value_sdpa.grad, rtol=1e-3, atol=1e-4)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        B, H, L, S, E = 2, 4, 8, 8, 16
        E_v = 16

        # Create inputs with large values
        query = torch.randn(B, H, L, E) * 10
        key = torch.randn(B, H, S, E) * 10
        value = torch.randn(B, H, S, E_v)

        # FlexAttention
        flex_output = flex_attention_decomposition(query, key, value)

        # SDPA
        sdpa_output = F.scaled_dot_product_attention(query, key, value)

        # Both should produce finite outputs
        assert torch.isfinite(flex_output).all()
        assert torch.isfinite(sdpa_output).all()

        # Outputs should be close
        torch.testing.assert_close(flex_output, sdpa_output, rtol=1e-3, atol=1e-4)

    def test_return_lse(self, basic_inputs):
        """Test returning log-sum-exp"""
        query, key, value = basic_inputs

        # FlexAttention with LSE
        output, lse = flex_attention_decomposition(query, key, value, return_lse=True)

        # Check shapes
        B, H, L, _ = query.shape
        _, _, _, E_v = value.shape
        assert output.shape == (B, H, L, E_v)
        assert lse.shape == (B, H, L)
        assert torch.isfinite(output).all()
        assert torch.isfinite(lse).all()

        # Verify LSE is computed correctly
        # LSE should be log(sum(exp(scores)))
        # For standard attention, this should be close to 0 after softmax
        # (since softmax normalizes to sum=1, log(1)=0)
        # But with masking, LSE can vary
        assert lse.abs().mean() < 10.0  # Reasonable range


def test_manual_comparison():
    """Manual comparison test for debugging"""
    print("\n" + "=" * 80)
    print("Manual Comparison: FlexAttention vs SDPA")
    print("=" * 80)

    B, H, L, E = 1, 2, 4, 8

    query = torch.randn(B, H, L, E)
    key = torch.randn(B, H, L, E)
    value = torch.randn(B, H, L, E)

    print("\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key:   {key.shape}")
    print(f"  Value: {value.shape}")

    # Test 1: Basic attention
    print(f"\n{'='*80}")
    print("Test 1: Basic Attention (no masking)")
    print("=" * 80)

    flex_output = flex_attention_decomposition(query, key, value)
    sdpa_output = F.scaled_dot_product_attention(query, key, value)

    print(f"FlexAttention output: mean={flex_output.mean():.6f}, std={flex_output.std():.6f}")
    print(f"SDPA output:          mean={sdpa_output.mean():.6f}, std={sdpa_output.std():.6f}")

    abs_diff = torch.abs(flex_output - sdpa_output)
    print(f"Absolute difference:  max={abs_diff.max():.6e}, mean={abs_diff.mean():.6e}")

    matches = torch.allclose(flex_output, sdpa_output, rtol=1e-4, atol=1e-5)
    print(f"Outputs match: {matches}")

    # Test 2: Causal attention
    print(f"\n{'='*80}")
    print("Test 2: Causal Attention")
    print("=" * 80)

    def causal_score_mod(score, batch, head, q_idx, k_idx):
        return torch.where(q_idx >= k_idx, score, float("-inf"))

    flex_causal = flex_attention_decomposition(query, key, value, score_mod=causal_score_mod)
    sdpa_causal = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    print(f"FlexAttention output: mean={flex_causal.mean():.6f}, std={flex_causal.std():.6f}")
    print(f"SDPA output:          mean={sdpa_causal.mean():.6f}, std={sdpa_causal.std():.6f}")

    abs_diff = torch.abs(flex_causal - sdpa_causal)
    print(f"Absolute difference:  max={abs_diff.max():.6e}, mean={abs_diff.mean():.6e}")

    matches = torch.allclose(flex_causal, sdpa_causal, rtol=1e-4, atol=1e-5)
    print(f"Outputs match: {matches}")

    # Test 3: Sliding window (FlexAttention only)
    print(f"\n{'='*80}")
    print("Test 3: Sliding Window Attention (FlexAttention only)")
    print("=" * 80)

    window_size = 2

    def sliding_window_score_mod(score, batch, head, q_idx, k_idx):
        causal_mask = q_idx >= k_idx
        window_mask = (q_idx - k_idx) <= window_size
        combined_mask = causal_mask & window_mask
        return torch.where(combined_mask, score, float("-inf"))

    flex_sliding = flex_attention_decomposition(
        query, key, value, score_mod=sliding_window_score_mod
    )

    print(f"FlexAttention output: mean={flex_sliding.mean():.6f}, std={flex_sliding.std():.6f}")
    print("Note: SDPA does not support sliding window natively")

    # Compare with full causal
    diff_from_causal = torch.abs(flex_sliding - flex_causal)
    print(
        f"Difference from full causal: max={diff_from_causal.max():.6e}, "
        f"mean={diff_from_causal.mean():.6e}"
    )
    print(f"Sliding window is more restrictive: {not torch.allclose(flex_sliding, flex_causal)}")

    print(f"\n{'='*80}")
    print("✅ All manual tests completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run manual comparison
    test_manual_comparison()

    # Run pytest tests
    print("\nRunning pytest tests...")
    print("=" * 80)

    test = TestFlexAttentionVsSDPA()

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

    B, H, L, S, E = 1, 4, 128, 128, 64
    long_inputs = (torch.randn(B, H, L, E), torch.randn(B, H, S, E), torch.randn(B, H, S, E))

    print("Test basic attention vs SDPA")
    test.test_basic_attention_vs_sdpa(basic_inputs)

    print("Test causal attention vs SDPA")
    test.test_causal_attention_vs_sdpa(basic_inputs)

    print("Test custom scale vs SDPA")
    test.test_custom_scale_vs_sdpa(basic_inputs)

    print("Test sliding window attention")
    test.test_sliding_window_attention(basic_inputs)

    print("Test GQA basic")
    test.test_gqa_basic(gqa_inputs)

    print("Test GQA with causal")
    test.test_gqa_with_causal(gqa_inputs)

    print("Test different seq lengths")
    test.test_different_seq_lengths()

    print("Test long sequences")
    test.test_long_sequences(long_inputs)

    print("Test gradient flow comparison")
    test.test_gradient_flow_comparison(basic_inputs)

    print("Test numerical stability")
    test.test_numerical_stability()

    print("Test return LSE")
    test.test_return_lse(basic_inputs)

    print("\n✅ All tests passed!")
