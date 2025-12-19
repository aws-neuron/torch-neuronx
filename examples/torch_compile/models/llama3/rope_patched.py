"""
Patched RoPE implementation for Neuron torch.compile backend.

This module provides a RoPE implementation that avoids complex number operations,
which are not supported by torch-mlir. The patched version is mathematically
equivalent to the original (verified with max difference < 5e-7).

Key changes:
1. precompute_freqs_cis_patched() returns separate cos/sin tensors instead of complex
2. apply_rotary_emb_patched() uses real number operations instead of complex multiplication

Usage:
    Replace the original RoPE functions in model.py with these patched versions.
"""

import torch


def precompute_freqs_cis_patched(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute RoPE frequencies as separate cos/sin tensors (no complex numbers).

    This is mathematically equivalent to the original precompute_freqs_cis but
    returns cos and sin separately instead of a complex tensor.

    Args:
        dim (int): Dimension of the frequency tensor (must be even).
        end (int): End index for precomputing frequencies (max sequence length).
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (cos_freqs, sin_freqs) each of shape [end, dim//2]
    """
    assert dim % 2 == 0, "dim must be even for RoPE"

    # Compute frequencies for each dimension pair
    dim_indices = torch.arange(0, dim, 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (dim_indices / float(dim)))

    # Create position indices
    positions = torch.arange(end, dtype=torch.float32)

    # Outer product: [end, dim//2]
    freqs = torch.outer(positions, freqs)

    # Return cos and sin separately
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)

    return cos_freqs, sin_freqs


def apply_rotary_emb_patched(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE using only real number operations (no complex numbers).

    This is mathematically equivalent to the original apply_rotary_emb but uses
    real number operations instead of complex multiplication.

    RoPE rotates pairs of dimensions: (x[0], x[1]), (x[2], x[3]), ...
    For each pair (a, b), rotation by angle θ gives:
        a' = a*cos(θ) - b*sin(θ)
        b' = a*sin(θ) + b*cos(θ)

    Args:
        xq (torch.Tensor): Query tensor [batch, seq, heads, head_dim]
        xk (torch.Tensor): Key tensor [batch, seq, heads, head_dim]
        cos (torch.Tensor): Cosine frequencies [max_seq, head_dim//2]
        sin (torch.Tensor): Sine frequencies [max_seq, head_dim//2]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Rotated (xq, xk) tensors with same shapes as inputs
    """
    # Split into even and odd indices along last dimension
    # x[..., 0::2] gets indices 0, 2, 4, ... (even)
    # x[..., 1::2] gets indices 1, 3, 5, ... (odd)
    xq_even = xq[..., 0::2]  # [batch, seq, heads, head_dim//2]
    xq_odd = xq[..., 1::2]  # [batch, seq, heads, head_dim//2]
    xk_even = xk[..., 0::2]
    xk_odd = xk[..., 1::2]

    # Slice cos/sin to actual sequence length and add broadcast dimensions
    seq_len = xq.shape[1]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]

    # Apply rotation
    # (a, b) -> (a*cos - b*sin, a*sin + b*cos)
    xq_even_rot = xq_even * cos - xq_odd * sin
    xq_odd_rot = xq_even * sin + xq_odd * cos
    xk_even_rot = xk_even * cos - xk_odd * sin
    xk_odd_rot = xk_even * sin + xk_odd * cos

    # Interleave back: stack and flatten
    # Stack: [batch, seq, heads, head_dim//2, 2]
    # Flatten: [batch, seq, heads, head_dim]
    xq_out = torch.stack([xq_even_rot, xq_odd_rot], dim=-1).flatten(-2, -1)
    xk_out = torch.stack([xk_even_rot, xk_odd_rot], dim=-1).flatten(-2, -1)

    return xq_out.type_as(xq), xk_out.type_as(xk)
