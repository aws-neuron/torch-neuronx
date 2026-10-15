"""
Native multi-head attention operation that orchestrates the three sub-operations:
- prefix: QKV projection and transformation
- mid: Attention computation
- suffix: Output projection
"""

import warnings

import torch

from .native_multi_head_attn_mid import NativeMultiHeadAttnMidOp
from .native_multi_head_attn_prefix import NativeMultiHeadAttnPrefixOp
from .native_multi_head_attn_suffix import NativeMultiHeadAttnSuffixOp

# Create singleton instances of the sub-operations
_prefix_op = NativeMultiHeadAttnPrefixOp()
_mid_op = NativeMultiHeadAttnMidOp()
_suffix_op = NativeMultiHeadAttnSuffixOp()


def native_multi_head_attention_neuron(
    query,
    key,
    value,
    embed_dim,
    num_head,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights=True,
    average_attn_weights=True,
    mask_type=None,
):
    """
    Implements PyTorch's _native_multi_head_attention for Neuron devices.

    This function orchestrates three sub-operations:
    1. prefix: QKV projection and transformation
    2. mid: Attention computation with optional causal masking
    3. suffix: Output projection back to model dimension

    Args:
        query: Query tensor of shape [batch, seq_len, embed_dim]
        key: Key tensor of shape [batch, seq_len, embed_dim]
        value: Value tensor of shape [batch, seq_len, embed_dim]
        embed_dim: Total embedding dimension
        num_head: Number of attention heads
        qkv_weight: Combined QKV weight matrix [3*embed_dim, embed_dim]
        qkv_bias: Combined QKV bias [3*embed_dim]
        proj_weight: Output projection weight [embed_dim, embed_dim]
        proj_bias: Output projection bias [embed_dim]
        mask: Optional attention mask (currently not fully supported)
        need_weights: Whether to return attention weights
        average_attn_weights: Whether to average attention weights across heads
        mask_type: Type of mask (0: attention mask, 1: causal mask)

    Returns:
        Tuple of (output, attn_weights) where:
        - output: Attention output of shape [batch, seq_len, embed_dim]
        - attn_weights: Empty tensor (attention weights not yet supported)

    Raises:
        NotImplementedError: For unsupported features
    """
    # Validate inputs
    if query.device.type != "neuron":
        raise ValueError(f"Expected Neuron device, got {query.device}")

    if embed_dim % num_head != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_head ({num_head})")

    # Check for unsupported features

    if mask is not None:
        # For now, we don't support arbitrary masks
        raise NotImplementedError(
            "Arbitrary mask tensors not yet supported in Neuron implementation. "
            "Please use mask_type=1 for causal masking or mask=None."
        )

    if need_weights and average_attn_weights is False:
        # We don't support returning per-head attention weights
        raise NotImplementedError(
            "Returning per-head attention weights not yet supported in Neuron implementation. "
            "Please use average_attn_weights=True or need_weights=False."
        )

    # Check sequence length constraint for flash attention
    seq_len = query.shape[1]
    if seq_len % 512 != 0:
        # Flash attention requires sequence length to be multiple of 512
        # Issue warning once per workload and continue with XLA fallback
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn(
                f"Sequence length {seq_len} is not a multiple of 512. "
                "Falling back to XLA/MLIR implementation instead of flash "
                "attention for better compatibility.",
                UserWarning,
                stacklevel=2,
            )
    # Part 1: QKV Projection and Transformation
    # Transforms inputs to multi-head format: [batch, num_heads, d_head, seq_len]
    q, k, v = _prefix_op(query, key, value, qkv_weight, qkv_bias, num_heads=num_head)

    # Part 2: Attention Computation
    # Computes scaled dot-product attention with optional causal masking
    use_causal_mask = False
    if mask_type is not None and mask_type == 1:
        use_causal_mask = True

    attn_output = _mid_op(
        q,
        k,
        v,
        use_causal_mask=use_causal_mask,
        dropout_p=0.0,  # No dropout for inference
        training=False,  # Inference mode
    )

    # Part 3: Output Projection
    # Transforms back from multi-head format and applies final projection
    output = _suffix_op(attn_output, proj_weight, proj_bias, num_heads=num_head)

    # Return attention weights
    if need_weights:
        # We don't support extracting attention weights from flash attention
        raise NotImplementedError(
            "Returning attention weights (need_weights=True) is not yet supported in Neuron "
            "implementation. Please use need_weights=False."
        )

    # Return empty tensor for attention weights when not needed (matches PyTorch behavior)
    attn_weights = torch.empty(0, device=query.device, dtype=query.dtype)

    return output, attn_weights


def native_multi_head_attention_out_neuron(
    query,
    key,
    value,
    embed_dim,
    num_head,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights=True,
    average_attn_weights=True,
    mask_type=None,
    out=None,
):
    """
    Out-variant of native_multi_head_attention that writes to pre-allocated tensors.

    Note: Currently, this implementation doesn't use the out parameter for the
    intermediate operations, as they allocate their own outputs. Future optimization
    could propagate the out parameter through the sub-operations.
    """
    # Compute the result using the standard function
    output, attn_weights = native_multi_head_attention_neuron(
        query,
        key,
        value,
        embed_dim,
        num_head,
        qkv_weight,
        qkv_bias,
        proj_weight,
        proj_bias,
        mask,
        need_weights,
        average_attn_weights,
        mask_type,
    )

    # Copy to the output tensor if provided
    if out is not None:
        if isinstance(out, tuple | list):
            # If out is a tuple, use the first element for output
            out[0].copy_(output)
            if len(out) > 1 and need_weights:
                out[1].copy_(attn_weights)
        else:
            # Single tensor provided
            out.copy_(output)
        return out

    return output, attn_weights
