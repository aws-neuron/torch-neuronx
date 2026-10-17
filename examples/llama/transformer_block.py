import torch
import torch.nn as nn
import torch.nn.functional as f
from self_attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """Apply root mean square normalization.

    Args:
        d_model (int): Size of the normalized final dimension.
        eps (float, optional): Numerical stability term. Defaults to ``1e-6``.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate RMS
        """Normalize and scale the final tensor dimension.

        Args:
            x (torch.Tensor): Input whose final dimension has size ``d_model``.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as ``x``.
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class FeedForward(nn.Module):
    """Implement a SwiGLU feed-forward network.

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Hidden projection dimension.
        dropout (float, optional): Hidden-state dropout probability. Defaults to ``0.0``.
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU activation
        """Apply the SwiGLU feed-forward transformation.

        Args:
            x (torch.Tensor): Input whose final dimension has size ``d_model``.

        Returns:
            torch.Tensor: Output with the same shape as ``x``.
        """
        gate = f.silu(self.w1(x))
        up = self.w3(x)
        x = gate * up
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerBlock(nn.Module):
    """Implement a pre-normalized transformer block.

    Args:
        d_model (int): Hidden dimension of the block.
        n_heads (int): Number of self-attention heads.
        d_ff (int): Hidden dimension of the feed-forward network.
        dropout (float, optional): Attention-weight, feed-forward hidden-state, and
            residual-branch dropout probability. Defaults to ``0.0``.
        use_rms_norm (bool, optional): Use RMS normalization instead of layer normalization.
            Defaults to ``True``.
    """

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
        """Apply self-attention and feed-forward residual layers.

        Args:
            x (torch.Tensor): Input with shape ``(batch_size, sequence_length, d_model)``.
            mask (torch.Tensor, optional): Additive mask broadcastable to attention scores.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Block output with the same shape as ``x``.
        """
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
    """Apply a sequence of transformer blocks.

    Args:
        n_layers (int): Number of transformer blocks.
        d_model (int): Hidden dimension of each block.
        n_heads (int): Number of self-attention heads per block.
        d_ff (int): Hidden dimension of each feed-forward network.
        dropout (float, optional): Attention-weight, feed-forward hidden-state, and
            residual-branch dropout probability. Defaults to ``0.0``.
        use_rms_norm (bool, optional): Use RMS normalization instead of layer normalization.
            Defaults to ``True``.
    """

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.0, use_rms_norm=True):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, use_rms_norm)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        """Apply every transformer block in sequence.

        Args:
            x (torch.Tensor): Input with shape ``(batch_size, sequence_length, d_model)``.
            mask (torch.Tensor, optional): Additive mask broadcastable to attention scores.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Stack output with the same shape as ``x``.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
