import torch
import torch.nn as nn

# ============================================================================
# TorchTitan Implementation
# Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/qwen3/model/model.py#L31-101
# ============================================================================


def precompute_rope_cache(dim: int, max_seq_len: int, base: float = 1_000_000.0) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, num_heads, head_dim]
    head_dim = xq.shape[-1]

    # reshape for broadcast
    rope_cache = reshape_for_broadcast(rope_cache, xq)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEBlock(nn.Module):
    def __init__(self, head_dim, max_seq_len, rope_theta):
        super().__init__()
        self.register_buffer(
            "rope_cache", precompute_rope_cache(head_dim, max_seq_len, rope_theta), persistent=False
        )

    def forward(self, xq, xk):
        return apply_rotary_emb(xq, xk, self.rope_cache)


# ============================================================================
# HuggingFace Implementation
# Reference: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py
# ============================================================================


def rotate_half_hf(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_hf(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine part [batch, seq_len, head_dim]
        sin: Sine part [batch, seq_len, head_dim]
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Tuple of rotated query and key tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_hf(q) * sin)
    k_embed = (k * cos) + (rotate_half_hf(k) * sin)
    return q_embed, k_embed


def precompute_rope_cache_hf(dim: int, max_seq_len: int, base: float = 1_000_000.0, device=None):
    """Precompute RoPE cache (HF style) - returns separate cos and sin tensors."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    freqs = position_ids @ inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


class RoPEBlockHF(nn.Module):
    """RoPE block using HuggingFace implementation.

    This expects input tensors in shape [batch, seq_len, num_heads, head_dim]
    and internally transposes to [batch, num_heads, seq_len, head_dim] for HF-style processing.
    """

    def __init__(self, head_dim, max_seq_len, rope_theta):
        super().__init__()
        cos, sin = precompute_rope_cache_hf(head_dim, max_seq_len, rope_theta)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, xq, xk):
        # Input: [batch, seq_len, num_heads, head_dim]
        # HF expects: [batch, num_heads, seq_len, head_dim]
        batch, seq_len, num_heads, head_dim = xq.shape

        # Transpose to HF format
        xq_t = xq.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        xk_t = xk.transpose(1, 2)

        # Get cos/sin for current sequence length
        cos = self.cos_cache[:seq_len].unsqueeze(0)  # [1, seq_len, head_dim]
        sin = self.sin_cache[:seq_len].unsqueeze(0)

        # Apply rotary embeddings
        xq_out, xk_out = apply_rotary_pos_emb_hf(xq_t, xk_t, cos, sin, unsqueeze_dim=1)

        # Transpose back to original format
        return xq_out.transpose(1, 2), xk_out.transpose(1, 2)
