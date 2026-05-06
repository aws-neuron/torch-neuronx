"""GPT-OSS Transformer Block definition for TP/FSDP benchmarking.

Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/gpt_oss/model/model.py

Adaptations from original:
- FlexAttention -> SDPA type attention
- MoE FFN -> SwiGLU FFN
- Removed attention sinks (don't affect sharding)

Model config from NeuronTorchTitan experiments/gpt_oss:
- dim=2880, n_heads=64, n_kv_heads=8, head_dim=64, hidden_dim=8640
- gpt-oss-20b: 24 layers
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from block_def.rope import apply_rotary_emb, precompute_rope_cache


@dataclass
class GptOssModelArgs:
    """GPT-OSS model arguments matching NeuronTorchTitan config."""

    dim: int = 2880
    n_heads: int = 64
    n_kv_heads: int = 8
    head_dim: int = 64
    hidden_dim: int = 8640
    n_layers: int = 1
    max_seq_len: int = 4096
    rope_theta: float = 150000.0
    norm_eps: float = 1e-5

    @classmethod
    def from_preset(cls, name: str):
        presets = {
            "gpt-oss-20b": cls(),  # 1 layer with 20B config
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name]


class Attention(nn.Module):
    """GQA attention."""

    def __init__(self, args: GptOssModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=True)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, rope_cache)

        q = q.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        v = v.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        # SDPA doesn't have enable_gqa, so repeat KV heads to match Q
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        output = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, args: GptOssModelArgs):
        super().__init__()
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rope_cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class GptOssTransformerBlock(nn.Module):
    """GPT-OSS transformer block wrapper with cached RoPE."""

    def __init__(self, args: GptOssModelArgs):
        super().__init__()
        self.block = TransformerBlock(args)
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(args.head_dim, args.max_seq_len, args.rope_theta),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x, self.rope_cache)
