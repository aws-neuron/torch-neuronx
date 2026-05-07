import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
        variance = x.pow(2).mean(dim=dims, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps)
        y = x * inv_rms
        if self.weight is not None:
            view_shape = (1,) * (x.dim() - len(self.normalized_shape)) + self.normalized_shape
            y = y * self.weight.view(view_shape)
        return y


# https://github.com/huggingface/transformers/blob/34e2c61259f8ba9dad64efe3e15d06c0772159f9/src/transformers/models/qwen3/modeling_qwen3.py#L49
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6, upcast: bool = True) -> None:
        """
        Qwen3RMSNorm with configurable upcast behavior
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.upcast = upcast

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32) if self.upcast else hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_dtype) if self.upcast else hidden_states
        return self.weight * hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, upcast={self.upcast}"
