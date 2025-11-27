"""
Tests for the complete _native_multi_head_attention operation.
"""

import pytest
import torch

from tests.utils.neuron_test_utils import assert_raises


def test_native_multi_head_attention_basic():
    """Test basic functionality of the complete multi-head attention operation."""
    torch.manual_seed(42)

    # Test parameters (matching constraints from sub-operations)
    batch = 2
    seq_len = 2048  # Minimum required for flash attention
    d_model = 1024
    num_heads = 8

    # Create inputs on Neuron device
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Call attention
    output, attn_weights = torch.ops.aten._native_multi_head_attention(
        query,
        key,
        value,
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=None,
        need_weights=False,
        mask_type=None,
    )

    # Verify output shape
    assert output.shape == (
        batch,
        seq_len,
        d_model,
    ), f"Expected shape {(batch, seq_len, d_model)}, got {output.shape}"
    assert output.device.type == "neuron", f"Expected Neuron device, got {output.device}"
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    # Attention weights should be empty when need_weights=False
    assert attn_weights.numel() == 0, "Expected empty attention weights"

    print("✓ Basic test passed")


def test_native_multi_head_attention_self_attention():
    """Test self-attention case where query == key == value."""
    torch.manual_seed(42)

    batch = 1
    seq_len = 2048
    d_model = 512
    num_heads = 8

    # Self-attention: same input for Q, K, V
    x = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Call with self-attention
    output, _ = torch.ops.aten._native_multi_head_attention(
        x,
        x,
        x,  # query == key == value
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=None,
        need_weights=False,
        mask_type=None,
    )

    assert output.shape == x.shape
    assert output.device == x.device
    assert output.dtype == x.dtype

    print("✓ Self-attention test passed")


def test_native_multi_head_attention_causal():
    """Test causal attention using mask_type parameter."""
    torch.manual_seed(42)

    batch = 2
    seq_len = 2048
    d_model = 768
    num_heads = 12

    # Create inputs
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Call with causal masking
    output, _ = torch.ops.aten._native_multi_head_attention(
        query,
        key,
        value,
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=None,
        need_weights=False,
        mask_type=1,  # Causal mask
    )

    assert output.shape == (batch, seq_len, d_model)
    assert output.device.type == "neuron"

    print("✓ Causal attention test passed")


def test_native_multi_head_attention_xla_fallback_short_sequence():
    """Test that short sequences issue a warning and use XLA fallback"""
    torch.manual_seed(42)

    batch = 1
    seq_len = 256  # Not a multiple of 512
    d_model = 256
    num_heads = 4

    # Create inputs on Neuron device
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Should issue a warning and succeed with XLA fallback
    with pytest.warns(UserWarning, match="Sequence length.*not a multiple of 512"):
        output, _ = torch.ops.aten._native_multi_head_attention(
            query,
            key,
            value,
            d_model,
            num_heads,
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            need_weights=False,
        )

    assert output.shape == (batch, seq_len, d_model)
    assert output.device.type == "neuron"
    print("✓ Short sequence warning test passed")


@assert_raises(NotImplementedError, match="need_weights=True.*not yet supported")
def test_native_multi_head_attention_not_implemented_need_weights():
    """Test that need_weights=True raises an error."""
    torch.manual_seed(42)

    batch = 1
    seq_len = 2048
    d_model = 512
    num_heads = 8

    # Create inputs
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # This should raise NotImplementedError when need_weights=True
    output, _ = torch.ops.aten._native_multi_head_attention(
        query,
        key,
        value,
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        need_weights=True,  # This triggers the error
    )

    print("✓ Need weights error test passed")


@assert_raises(NotImplementedError, match="Arbitrary mask tensors not yet supported")
def test_native_multi_head_attention_not_implemented_mask():
    """Test that arbitrary masks raise an error"""
    torch.manual_seed(42)

    batch = 1
    seq_len = 2048
    d_model = 512
    num_heads = 8

    # Create inputs
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Create an arbitrary mask
    mask = torch.ones(batch, seq_len, seq_len, device="neuron", dtype=torch.bool)

    # This should raise NotImplementedError due to unsupported mask
    output, _ = torch.ops.aten._native_multi_head_attention(
        query,
        key,
        value,
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=mask,
    )

    print("✓ Mask error test passed")


@pytest.mark.parametrize(
    "batch,seq_len,d_model,num_heads",
    [
        (1, 512, 256, 4),  # Small sequence
        (1, 1024, 512, 8),  # Medium sequence
        (2, 2048, 512, 8),  # Default tile size
        (1, 4096, 1024, 16),  # Large sequence
    ],
    ids=[
        "batch_1_512_tokens_256_embed_4_heads",
        "batch_1_1024_tokens_512_embed_8_heads",
        "batch_2_2048_tokens_512_embed_8_heads",
        "batch_1_4096_tokens_1024_embed_16_heads",
    ],
)
def test_native_multi_head_attention_different_dimensions(batch, seq_len, d_model, num_heads):
    """Test with different dimension configurations."""
    torch.manual_seed(42)

    # Create inputs
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Call the operation
    output, _ = torch.ops.aten._native_multi_head_attention(
        query,
        key,
        value,
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=None,
        need_weights=False,
    )

    assert output.shape == (batch, seq_len, d_model)
    assert output.device.type == "neuron"

    print(
        f"✓ Test passed for config: batch={batch}, seq_len={seq_len}, "
        f"d_model={d_model}, num_heads={num_heads}"
    )


def test_native_multi_head_attention_neuron_execution():
    """Verify that _native_multi_head_attention executes on Neuron, not CPU."""
    import os

    torch.manual_seed(42)

    # Get current PID for log file
    pid = os.getpid()
    log_dir = os.path.join(os.getcwd(), ".torch_neuronx", "offloaded_ops")
    log_file = os.path.join(log_dir, f"{pid}.txt")

    # Test parameters
    batch = 1
    seq_len = 2048
    d_model = 512
    num_heads = 8

    # Create inputs on Neuron device
    query = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    key = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)
    value = torch.randn(batch, seq_len, d_model, device="neuron", dtype=torch.float32)

    # Create weight matrices
    qkv_weight = torch.randn(3 * d_model, d_model, device="neuron", dtype=torch.float32)
    qkv_bias = torch.randn(3 * d_model, device="neuron", dtype=torch.float32)
    proj_weight = torch.randn(d_model, d_model, device="neuron", dtype=torch.float32)
    proj_bias = torch.randn(d_model, device="neuron", dtype=torch.float32)

    # Call attention - this should run on Neuron, not CPU
    output, _ = torch.ops.aten._native_multi_head_attention(
        query,
        key,
        value,
        d_model,
        num_heads,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask=None,
        need_weights=False,
    )

    # Check if our specific op was offloaded to CPU
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_content = f.read()
        # Check if _native_multi_head_attention appears in the log
        if "_native_multi_head_attention" in log_content:
            raise AssertionError(
                f"_native_multi_head_attention fell back to CPU!\n"
                f"Found in offloaded operations log:\n{log_content}"
            )

    # Verify the result is on Neuron device
    assert output.device.type == "neuron", f"Expected Neuron device, got {output.device}"
    assert output.shape == (batch, seq_len, d_model)

    print("✓ Neuron execution verification passed")


if __name__ == "__main__":
    # Import torch_neuronx to register the device

    # Run all tests
    test_native_multi_head_attention_basic()
    test_native_multi_head_attention_self_attention()
    test_native_multi_head_attention_causal()
    test_native_multi_head_attention_xla_fallback_short_sequence()
    test_native_multi_head_attention_not_implemented_need_weights()
    test_native_multi_head_attention_not_implemented_mask()
    test_native_multi_head_attention_different_dimensions()
    test_native_multi_head_attention_neuron_execution()

    print("\n✅ All tests passed!")
