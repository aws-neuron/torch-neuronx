import math

import torch
import torch.nn as nn
import torch.nn.functional as f


def create_causal_mask(seq_length, device):
    """Create a causal mask for self-attention."""
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism used in transformers."""

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
