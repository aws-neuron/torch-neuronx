import pytest
import torch

from torch_neuronx.python_ops.native_multi_head_attn_mid import NativeMultiHeadAttnMidOp
from torch_neuronx.python_ops.native_multi_head_attn_prefix import NativeMultiHeadAttnPrefixOp
from torch_neuronx.python_ops.native_multi_head_attn_suffix import NativeMultiHeadAttnSuffixOp

# Create module-level instances of the operations for testing
_suffix_op = NativeMultiHeadAttnSuffixOp()
_prefix_op = NativeMultiHeadAttnPrefixOp()
_mid_op = NativeMultiHeadAttnMidOp()


def test_suffix_basic():
    """Test basic suffix operation with identity projection"""
    batch, num_heads, seq_len, d_head = 2, 8, 2048, 128
    d_model = num_heads * d_head

    # Create input
    torch.manual_seed(42)
    attn_output = torch.randn(batch, num_heads, seq_len, d_head).to("neuron")

    # Use identity projection to test reshape
    proj_weight = torch.eye(d_model).to("neuron")
    proj_bias = torch.zeros(d_model).to("neuron")

    # Run suffix operation
    result = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_heads)

    # Verify output shape
    assert result.shape == (batch, seq_len, d_model)

    # With identity matrix, output should equal reshaped input
    expected = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
    print("✓ Basic suffix test passed")


def test_suffix_projection():
    """Test suffix operation with non-identity projection"""
    batch, num_heads, seq_len, d_head = 2, 8, 2048, 128
    d_model = num_heads * d_head

    # Create input
    torch.manual_seed(42)
    attn_output = torch.randn(batch, num_heads, seq_len, d_head).to("neuron")

    # Use random projection matrix
    proj_weight = torch.randn(d_model, d_model).to("neuron")
    proj_bias = torch.randn(d_model).to("neuron")

    # Run suffix operation
    result = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_heads)

    # Verify output shape
    assert result.shape == (batch, seq_len, d_model)

    # Manually compute expected result
    reshaped = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    expected = reshaped @ proj_weight.T + proj_bias

    torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)
    print("✓ Projection test passed")


def test_suffix_out_variant():
    """Test out-variant of suffix operation"""
    batch, num_heads, seq_len, d_head = 2, 8, 2048, 128
    d_model = num_heads * d_head

    # Create input
    torch.manual_seed(42)
    attn_output = torch.randn(batch, num_heads, seq_len, d_head).to("neuron")
    proj_weight = torch.randn(d_model, d_model).to("neuron")
    proj_bias = torch.randn(d_model).to("neuron")

    # Pre-allocate output
    out = torch.empty(batch, seq_len, d_model).to("neuron")

    # Run out-variant
    result = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_heads, out=out)

    # Verify result is the same tensor as out
    assert result is out
    assert result.shape == (batch, seq_len, d_model)

    # Compare with regular variant
    expected = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_heads)

    torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
    print("✓ Out-variant test passed")


def test_full_integration():
    """Test complete multi-head attention pipeline"""
    batch, seq_len, d_model, num_heads = 2, 2048, 1024, 8

    # Setup inputs
    torch.manual_seed(42)
    query = torch.randn(batch, seq_len, d_model).to("neuron")
    key = query.clone()  # Self-attention
    value = query.clone()

    qkv_weight = torch.randn(3 * d_model, d_model).to("neuron")
    qkv_bias = torch.randn(3 * d_model).to("neuron")
    proj_weight = torch.randn(d_model, d_model).to("neuron")
    proj_bias = torch.randn(d_model).to("neuron")

    # Run through all three parts
    q, k, v = _prefix_op(query, key, value, qkv_weight, qkv_bias, num_heads=num_heads)

    attn_output = _mid_op(q, k, v, use_causal_mask=False)

    final_output = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_heads)

    # Verify final shape
    assert final_output.shape == (batch, seq_len, d_model)
    print("✓ Integration test passed")


@pytest.mark.parametrize(
    "batch,num_heads,seq_len,d_head",
    [
        (1, 4, 2048, 64),  # Single batch, small heads
        (4, 16, 2048, 64),  # Larger batch, more heads
        (2, 8, 4096, 128),  # Longer sequence
    ],
    ids=[
        "batch_1_4_heads_2048_tokens_64_dim",
        "batch_4_16_heads_2048_tokens_64_dim",
        "batch_2_8_heads_4096_tokens_128_dim",
    ],
)
def test_different_dimensions(batch, num_heads, seq_len, d_head):
    """Test with different dimension configurations"""
    d_model = num_heads * d_head

    torch.manual_seed(42)
    attn_output = torch.randn(batch, num_heads, seq_len, d_head).to("neuron")
    proj_weight = torch.randn(d_model, d_model).to("neuron")
    proj_bias = torch.randn(d_model).to("neuron")

    result = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_heads)

    assert result.shape == (batch, seq_len, d_model)

    print("✓ Different dimensions test passed")


if __name__ == "__main__":
    test_suffix_basic()
    test_suffix_projection()
    test_suffix_out_variant()
    test_full_integration()
    test_different_dimensions()
    print("\n✓ All tests passed!")
