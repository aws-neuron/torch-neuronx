# block_def/qwen3_torchtitan.py
"""
TorchTitan Qwen3 Model Definition
Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/qwen3/model/model.py
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as func

from .rope import apply_rotary_emb, precompute_rope_cache


@dataclass
class Qwen3ModelArgs:
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072
    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True
    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"
    eos_id: int = 151645
    enable_weight_tying: bool = False
    moe_enabled: bool = False
    moe_inter_dim: int = 768


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim

        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor, attention_masks=None):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, rope_cache)
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        output = func.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(func.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(model_args.dim, model_args.hidden_dim)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor, attention_masks=None):
        x = x + self.attention(self.attention_norm(x), rope_cache, attention_masks)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Qwen3TransformerBlock(nn.Module):
    """Single Qwen3 transformer block for testing."""

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.block = TransformerBlock(model_args)
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(
                model_args.head_dim, model_args.max_seq_len, model_args.rope_theta
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        return self.block(x, self.rope_cache, None)
