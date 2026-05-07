import torch
import torch.nn as nn
import torch.nn.functional as f
from self_attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU activation
        gate = f.silu(self.w1(x))
        up = self.w3(x)
        x = gate * up
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_rms_norm=True):
        super().__init__()

        # Choose normalization
        if use_rms_norm:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        else:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward with residual connection
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x


class TransformerStack(nn.Module):
    """Stack of transformer blocks."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.0, use_rms_norm=True):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, use_rms_norm)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
