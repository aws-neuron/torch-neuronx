from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .feedforward import FeedForward
from .rms_norm import RMSNorm


@dataclass
class LlamaConfig:
    """Configuration for Llama transformer block architecture."""

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int | None = None
    head_dim: int | None = None

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
            "llama-7b": cls(hidden_size=4096, num_attention_heads=32, head_dim=128),
            "llama-13b": cls(hidden_size=5120, num_attention_heads=40, head_dim=128),
            "llama-30b": cls(hidden_size=6656, num_attention_heads=52, head_dim=128),
            "llama-70b": cls(hidden_size=8192, num_attention_heads=64, head_dim=128),
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        return presets[preset_name]


class LlamaTransformerBlock(nn.Module):
    """Llama-style transformer block with RMSNorm and SwiGLU."""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        # RMS normalization layers that avoid dtype upcasts on Neuron
        self.ln1 = RMSNorm(hidden_size, eps=1e-6)
        self.ln2 = RMSNorm(hidden_size, eps=1e-6)

        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Feed-forward network
        self.feed_forward = FeedForward(hidden_size, intermediate_size)

    def forward(self, x, is_causal=True):
        batch_size, seq_len, _ = x.shape

        # Self-attention with residual connection
        residual = x
        x_norm = self.ln1(x)

        # Compute Q, K, V
        q = (
            self.q_proj(x_norm)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x_norm)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x_norm)
            .view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)

        # Reshape and project
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        x = residual + attn_output

        # Feed-forward with residual connection
        residual = x
        x_norm = self.ln2(x)
        ff_output = self.feed_forward(x_norm)
        x = residual + ff_output

        return x
