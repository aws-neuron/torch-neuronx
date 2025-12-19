from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .feedforward import FeedForward
from .rms_norm import RMSNorm
from .rope import apply_rotary_emb, precompute_rope_cache


@dataclass
class LlamaConfig:
    """Configuration for Llama transformer block architecture."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int | None = None
    head_dim: int | None = None
    max_seq_len: int = 4096
    rope_theta: float = 1_000_000.0

    def __post_init__(self):
        """Calculate derived values."""
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        else:
            # Ensure num_attention_heads is consistent with hidden_size and head_dim
            self.num_attention_heads = self.hidden_size // self.head_dim

    @classmethod
    def get_default(cls):
        """Get default configuration."""
        return cls(hidden_size=4096, num_attention_heads=32, head_dim=128)

    @classmethod
    def get_preset(cls, preset_name: str):
        """Get preset configurations for common Llama model sizes."""
        presets = {
            "llama-7b": cls(
                hidden_size=4096,
                num_attention_heads=32,
                head_dim=128,
                max_seq_len=4096,
                rope_theta=1_000_000.0,
            ),
            "llama-13b": cls(
                hidden_size=5120,
                num_attention_heads=40,
                head_dim=128,
                max_seq_len=4096,
                rope_theta=1_000_000.0,
            ),
            "llama-30b": cls(
                hidden_size=6656,
                num_attention_heads=52,
                head_dim=128,
                max_seq_len=4096,
                rope_theta=1_000_000.0,
            ),
            "llama-70b": cls(
                hidden_size=8192,
                num_attention_heads=64,
                head_dim=128,
                max_seq_len=4096,
                rope_theta=1_000_000.0,
            ),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        return presets[preset_name]


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_seq_len, rope_theta):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Precompute RoPE cache using TorchTitan implementation
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(self.head_dim, max_seq_len, rope_theta),
            persistent=False,
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        # Apply TorchTitan RoPE
        q, k = apply_rotary_emb(q, k, self.rope_cache)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)

        # Reshape and project
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaTransformerBlock(nn.Module):
    """Llama-style transformer block with RMSNorm, SwiGLU, and TorchTitan RoPE."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        max_seq_len=4096,
        rope_theta=1_000_000.0,
    ):
        super().__init__()

        # RMS normalization layers that avoid dtype upcasts on Neuron
        self.ln1 = RMSNorm(hidden_size, eps=1e-6)
        self.ln2 = RMSNorm(hidden_size, eps=1e-6)

        # Multi-head attention with TorchTitan RoPE
        self.attention = Attention(hidden_size, num_attention_heads, max_seq_len, rope_theta)

        # Feed-forward network
        self.feed_forward = FeedForward(hidden_size, intermediate_size)

    def forward(self, x, is_causal=True):
        # Self-attention with residual connection
        residual = x
        x_norm = self.ln1(x)

        attn_output = self.attention(x_norm)

        x = residual + attn_output

        # Feed-forward with residual connection
        residual = x
        x_norm = self.ln2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + ff_output

        return x
