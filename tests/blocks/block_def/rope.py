import torch
import torch.nn as nn

# ============================================================================
# TorchTitan Implementation
# Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/qwen3/model/model.py#L31-101
# ============================================================================


def precompute_rope_cache(dim: int, max_seq_len: int, base: float = 1_000_000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(xq, xk, rope_cache):
    head_dim = xq.shape[-1]
    seqlen = xq.shape[1]
    rope_cache = rope_cache[0:seqlen].view(-1, seqlen, 1, head_dim * 2)
    cos, sin = rope_cache[..., :head_dim], rope_cache[..., head_dim:]
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)


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
