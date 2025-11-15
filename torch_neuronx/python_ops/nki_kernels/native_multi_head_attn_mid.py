"""NKI kernel for native multi-head attention middle operation (self-attention computation)"""

import math

from neuronxcc.nki.kernels.attention import FlashConfig, flash_fwd


def native_multi_head_attn_mid_kernel_with_grid(
    q, k, v, attn_output, use_causal_mask=False, dropout_p=0.0, training=False
):
    """
    Alternative implementation using TorchNeuronNKIKernel with explicit grid.

    This version is closer to the mha.py implementation but wrapped properly
    for integration as an aten operation.
    """
    # Import here to avoid circular import
    from torch_neuronx import TorchNeuronNKIKernel

    batch_size, num_heads, d_head, seq_len = q.shape

    # Calculate softmax scale
    softmax_scale = 1.0 / math.sqrt(d_head)

    # Dynamically determine seq_tile_size based on sequence length
    # Get default from FlashConfig
    default_seq_tile_size = FlashConfig().seq_tile_size

    # Prefer default tile size when possible for optimal memory usage
    if seq_len % default_seq_tile_size == 0:
        # Keep the default - this is the preferred case
        seq_tile_size = default_seq_tile_size
    else:
        # Only change if we must - prefer larger tiles up to default for better perf
        # We know this works due to our can_handle check (seq_len % 512 == 0)
        seq_tile_size = 1024 if seq_len % 1024 == 0 else 512

    # Create config with custom seq_tile_size
    # Set should_transpose_v=True because our V has shape (batch, num_heads, d_head, seq_len)
    config = FlashConfig(training=training, should_transpose_v=True, seq_tile_size=seq_tile_size)

    # For inference, we don't need LSE or seed
    seed = None
    lse = None

    # Create the kernel with grid
    grid = (batch_size, num_heads)

    return_overrides = (attn_output, lse) if training else attn_output

    traced_kernel = TorchNeuronNKIKernel(
        func=getattr(flash_fwd, "func", flash_fwd),
        grid=grid,
        kernel_return=True,
        return_tensor_overrides=return_overrides,
    )

    if training:
        out_ret, _lse_ret = traced_kernel(
            q,
            k,
            v,
            seed,
            use_causal_mask=use_causal_mask,
            mixed_precision=True,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            config=config,
        )
        if out_ret is not attn_output:
            attn_output.copy_(out_ret)
    else:
        out_ret = traced_kernel(
            q,
            k,
            v,
            seed,
            use_causal_mask=use_causal_mask,
            mixed_precision=True,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            config=config,
        )
        if out_ret is not attn_output:
            attn_output.copy_(out_ret)

    return attn_output
