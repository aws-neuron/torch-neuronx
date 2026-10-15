import math

import torch
import torch.nn as nn
import torch.nn.functional as f


def create_causal_mask(seq_length, device):
    """Create an additive causal self-attention mask.

    Args:
        seq_length (int): Sequence length for both attention dimensions.
        device (torch.device or str): Device on which to create the mask.

    Returns:
        torch.Tensor: Mask with shape ``(1, 1, seq_length, seq_length)`` containing zero on
            and below the diagonal and negative infinity above it.
    """
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Implement multi-head self-attention.

    Args:
        d_model (int): Hidden dimension of the input and output.
        n_heads (int): Number of attention heads.
        dropout (float, optional): Attention-weight dropout probability. Defaults to ``0.0``.

    Raises:
        AssertionError: If ``d_model`` is not divisible by ``n_heads``.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """Apply multi-head self-attention.

        Args:
            x (torch.Tensor): Input with shape ``(batch_size, sequence_length, d_model)``.
            mask (torch.Tensor, optional): Additive mask broadcastable to attention scores.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Attention output with shape
                ``(batch_size, sequence_length, d_model)``.
        """
        batch_size, seq_length, d_model = x.shape

        # Linear transformations and reshape
        q = self.W_q(x).view(batch_size, seq_length, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_length, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_length, self.n_heads, self.d_k)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch, n_heads, seq_length, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores + mask

        attention_weights = f.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)

        # Final linear transformation
        output = self.W_o(context)

        return output
