import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        x = gate * up
        x = self.w2(x)
        return x
