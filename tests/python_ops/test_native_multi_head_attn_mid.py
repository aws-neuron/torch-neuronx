"""Test cases for native multi-head attention middle operation."""

import math
from unittest.mock import patch

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import (
    assert_op_runs_on_neuron,
    assert_raises,
    track_neuron_ops,
)
from torch_neuronx.python_ops.native_multi_head_attn_mid import NativeMultiHeadAttnMidOp
from torch_neuronx.python_ops.native_multi_head_attn_prefix import NativeMultiHeadAttnPrefixOp

# Create module-level instances of the operations for testing
_mid_op = NativeMultiHeadAttnMidOp()
_prefix_op = NativeMultiHeadAttnPrefixOp()


class TestNativeMultiHeadAttnMid:
    """Test cases for native multi-head attention middle operation."""

    def test_basic_attention(self):
        """Test basic attention computation without masking."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        # Create input tensors - all have the same shape format
        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Execute attention
        attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)

        # Verify output shape
        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
        assert attn_output.device.type == "neuron"
        assert not torch.isnan(attn_output).any()

    def test_causal_attention(self):
        """Test attention with causal masking."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Execute with causal mask
        attn_output = _mid_op(q, k, v, use_causal_mask=True, dropout_p=0.0, training=False)

        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
        assert not torch.isnan(attn_output).any()

    def test_out_variant(self):
        """Test the out variant of the operation."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Pre-allocate output tensor
        out = torch.empty(batch_size, num_heads, seq_len, d_head, device="neuron")
        out_ptr = out.data_ptr()

        # Execute with output tensor
        result = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False, out=out)

        # Verify in-place modification
        assert result is out
        assert out.data_ptr() == out_ptr
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        seq_len = 2048
        num_heads = 8
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        with track_neuron_ops():
            attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")

        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)

    @pytest.mark.parametrize("num_heads", [4, 8, 16])
    def test_different_head_counts(self, num_heads):
        """Test with different numbers of attention heads."""
        batch_size = 1
        seq_len = 2048
        d_model = 512  # Reduced to ensure d_head <= 128
        d_head = d_model // num_heads

        # Skip if d_head > 128 (kernel limitation)
        if d_head > 128:
            pytest.skip(f"d_head={d_head} exceeds kernel limitation of 128")

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        with track_neuron_ops():
            attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")

        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)

    def test_integration_with_prefix(self):
        """Test full pipeline with prefix operation."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_model = 1024
        d_head = d_model // num_heads

        # Create inputs for prefix operation
        query = torch.randn(batch_size, seq_len, d_model, device="neuron")
        key = torch.randn(batch_size, seq_len, d_model, device="neuron")
        value = torch.randn(batch_size, seq_len, d_model, device="neuron")
        qkv_weight = torch.randn(3 * d_model, d_model, device="neuron")
        qkv_bias = torch.randn(3 * d_model, device="neuron")

        # Apply prefix operation
        q, k, v = _prefix_op(query, key, value, qkv_weight, qkv_bias, num_heads=num_heads)

        # Verify prefix output shapes
        assert q.shape == (batch_size, num_heads, d_head, seq_len)
        assert k.shape == (batch_size, num_heads, d_head, seq_len)
        assert v.shape == (batch_size, num_heads, d_head, seq_len)

        # Apply middle operation - v is already in the correct format
        with track_neuron_ops():
            attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")

        # Verify final output
        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
        assert not torch.isnan(attn_output).any()

    @assert_raises(RuntimeError)
    def test_invalid_key_shape(self):
        """Test error handling for invalid key shape."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        # Valid tensors
        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Test with wrong key shape
        k_wrong = torch.randn(batch_size, num_heads, d_head, seq_len // 2, device="neuron")
        _mid_op(q, k_wrong, v)

    @assert_raises(RuntimeError)
    def test_invalid_value_shape_seq_len(self):
        """Test error handling for invalid value shape (different seq_len)."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        # Valid tensors
        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Test with wrong value shape (different seq_len)
        v_wrong = torch.randn(batch_size, num_heads, d_head, seq_len // 2, device="neuron")
        _mid_op(q, k, v_wrong)

    @assert_raises(RuntimeError)
    def test_invalid_value_shape_dimension_order(self):
        """Test error handling for invalid value shape (incorrect dimension order)."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        # Valid tensors
        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Test with wrong value shape (incorrect dimension order)
        v_wrong_format = torch.randn(batch_size, num_heads, seq_len, d_head, device="neuron")
        _mid_op(q, k, v_wrong_format)

    @pytest.mark.parametrize("seq_len", [2048, 4096, 8192])
    def test_default_seq_tile_size_multiples(self, seq_len):
        """Test that multiples of 2048 work with default seq_tile_size."""
        batch_size = 1
        num_heads = 8
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Should work with default config (seq_tile_size=2048)
        with track_neuron_ops():
            attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")

        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
        assert not torch.isnan(attn_output).any()

    @pytest.mark.parametrize("seq_len", [512, 1024, 1536])
    def test_smaller_sequences_with_dynamic_tile_size(self, seq_len):
        """Test that smaller sequences work with dynamically adjusted seq_tile_size."""
        batch_size = 1
        num_heads = 8
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Should work with dynamically adjusted seq_tile_size
        with track_neuron_ops():
            attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")

        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
        assert not torch.isnan(attn_output).any()

    def test_sequence_length_constraint_xla_fallback(self):
        """Test that NKI can't handle non-512 multiples but XLA fallback works."""
        batch_size = 1
        num_heads = 8
        d_head = 128
        seq_len = 768  # Not a multiple of 512

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Get the implementation instances from the operation
        nki_impl = _mid_op._implementations[0]  # First implementation is NKI
        xla_impl = _mid_op._implementations[1]  # Second implementation is XLA
        with (
            patch.object(nki_impl, "can_handle", wraps=nki_impl.can_handle) as mock_nki_can_handle,
            patch.object(xla_impl, "can_handle", wraps=xla_impl.can_handle) as mock_xla_can_handle,
            track_neuron_ops(),
        ):
            # Should succeed with XLA fallback
            attn_output = _mid_op(q, k, v)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")
            assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
            mock_nki_can_handle.assert_called()
            mock_xla_can_handle.assert_called()

    def test_sequence_too_small_xla_fallback(self):
        """Test that NKI can't handle small sequences but XLA fallback works."""
        batch_size = 1
        num_heads = 8
        d_head = 128
        seq_len = 256  # Too small for NKI

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Get the implementation instances from the operation
        nki_impl = _mid_op._implementations[0]  # First implementation is NKI
        xla_impl = _mid_op._implementations[1]  # Second implementation is XLA
        with (
            patch.object(nki_impl, "can_handle", wraps=nki_impl.can_handle) as mock_nki_can_handle,
            patch.object(xla_impl, "can_handle", wraps=xla_impl.can_handle) as mock_xla_can_handle,
            track_neuron_ops(),
        ):
            # Should succeed with XLA fallback
            attn_output = _mid_op(q, k, v)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")
            assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)
            mock_nki_can_handle.assert_called()
            mock_xla_can_handle.assert_called()

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["float32", "float16", "bfloat16"],
    )
    def test_different_dtypes(self, dtype):
        """Test with different data types."""
        batch_size = 1
        seq_len = 2048
        num_heads = 8
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron", dtype=dtype)
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron", dtype=dtype)
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron", dtype=dtype)

        with track_neuron_ops():
            attn_output = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)
            assert_op_runs_on_neuron("native_multi_head_attn_mid")

        assert attn_output.dtype == dtype
        assert attn_output.shape == (batch_size, num_heads, seq_len, d_head)

    def test_mid_attn_ir_cache_correctness(self):
        """Test that IR cache is properly incremented and hit on subsequent calls."""
        batch_size = 2
        num_heads = 4
        seq_len = 2048
        d_head = 128

        q = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        # Clear caches to ensure clean state
        torch_neuronx.nki_kernel.TorchNeuronNKIKernel.clear_all_caches()

        # Get initial cache size (should be 0)
        initial_cache_size = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert initial_cache_size == 0, f"Expected empty cache, got size {initial_cache_size}"

        # First call - should trace and cache IR
        output1 = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)

        # Cache should have 1 entry now
        cache_size_after_first = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert (
            cache_size_after_first == 1
        ), f"Expected cache size 1 after first call, got {cache_size_after_first}"

        # Second call - should use cached IR (cache size unchanged)
        output2 = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)

        cache_size_after_second = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert (
            cache_size_after_second == 1
        ), f"Expected cache size 1 after second call (cache hit), got {cache_size_after_second}"

        # Third call - should also use cached IR (cache size unchanged)
        output3 = _mid_op(q, k, v, use_causal_mask=False, dropout_p=0.0, training=False)

        cache_size_after_third = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert (
            cache_size_after_third == 1
        ), f"Expected cache size 1 after third call (cache hit), got {cache_size_after_third}"

        # Verify outputs are consistent
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(output1, output3, rtol=1e-5, atol=1e-5)

    def test_mid_attn_ir_cache_different_inputs_same_shape(self):
        """Test IR cache hit with different inputs, same shapes."""
        batch_size = 2
        num_heads = 4
        seq_len = 2048
        d_head = 128

        # Clear caches
        torch_neuronx.nki_kernel.TorchNeuronNKIKernel.clear_all_caches()

        # First call with specific input values
        torch.manual_seed(42)
        q1 = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k1 = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v1 = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        output1 = _mid_op(q1, k1, v1, use_causal_mask=False, dropout_p=0.0, training=False)

        # Second call with different input values but same shapes - should hit cache
        torch.manual_seed(123)
        q2 = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        k2 = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")
        v2 = torch.randn(batch_size, num_heads, d_head, seq_len, device="neuron")

        output2 = _mid_op(q2, k2, v2, use_causal_mask=False, dropout_p=0.0, training=False)

        # Verify shapes are the same
        assert output1.shape == output2.shape

        # Verify outputs are different (since inputs are different)
        # Use a threshold to ensure they're meaningfully different
        diff = torch.abs(output1 - output2).mean().item()
        assert diff > 1e-3, f"Outputs should be different with different inputs, but diff={diff}"

        print(
            f"Mid Attention IR Cache with different inputs: outputs differ by {diff:.6f} (expected)"
        )

    def test_mid_attn_ir_cache_different_shapes(self):
        """Test that IR cache creates separate entries for different shapes and different inputs."""
        batch_size = 2
        num_heads = 4
        d_head = 128

        # Clear caches
        torch_neuronx.nki_kernel.TorchNeuronNKIKernel.clear_all_caches()

        # First shape: seq_len = 2048
        seq_len1 = 2048
        q1 = torch.randn(batch_size, num_heads, d_head, seq_len1, device="neuron")
        k1 = torch.randn(batch_size, num_heads, d_head, seq_len1, device="neuron")
        v1 = torch.randn(batch_size, num_heads, d_head, seq_len1, device="neuron")

        output1 = _mid_op(q1, k1, v1, use_causal_mask=False, dropout_p=0.0, training=False)

        # Cache should have 1 entry
        cache_size_after_first = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert (
            cache_size_after_first == 1
        ), f"Expected cache size 1 after first shape, got {cache_size_after_first}"

        # Second shape: seq_len = 4096 (different shape)
        seq_len2 = 4096
        q2 = torch.randn(batch_size, num_heads, d_head, seq_len2, device="neuron")
        k2 = torch.randn(batch_size, num_heads, d_head, seq_len2, device="neuron")
        v2 = torch.randn(batch_size, num_heads, d_head, seq_len2, device="neuron")

        output2 = _mid_op(q2, k2, v2, use_causal_mask=False, dropout_p=0.0, training=False)

        # Cache should have 2 entries now (different shape creates new entry)
        cache_size_after_second = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert (
            cache_size_after_second == 2
        ), f"Expected cache size 2 after second shape, got {cache_size_after_second}"

        # Reuse first shape with different inputs - should hit cache (size stays at 2)
        q3 = torch.randn(batch_size, num_heads, d_head, seq_len1, device="neuron")
        k3 = torch.randn(batch_size, num_heads, d_head, seq_len1, device="neuron")
        v3 = torch.randn(batch_size, num_heads, d_head, seq_len1, device="neuron")

        output3 = _mid_op(q3, k3, v3, use_causal_mask=False, dropout_p=0.0, training=False)

        # Cache should still have 2 entries (cache hit for first shape)
        cache_size_after_third = torch_neuronx.nki_kernel.TorchNeuronNKIKernel.get_ir_cache_size()
        assert (
            cache_size_after_third == 2
        ), f"Expected cache size 2 after reusing first shape, got {cache_size_after_third}"

        # Verify output shapes
        assert output1.shape == (batch_size, num_heads, seq_len1, d_head)
        assert output2.shape == (batch_size, num_heads, seq_len2, d_head)
        assert output3.shape == (batch_size, num_heads, seq_len1, d_head)
