"""Mixtral MoE (8x7B) block definition.

Mixtral-8x7B: Llama-style GQA attention + top-2 routed SwiGLU expert FFNs.
Reference: https://arxiv.org/abs/2401.04088

# MFU baseline: 10.59% from first TRN2 run (2026-03-01).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .rms_norm import RMSNorm
from .rope import apply_rotary_emb, precompute_rope_cache


@dataclass
class MixtralConfig:
    """Configuration for Mixtral MoE transformer block."""

    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 14336
    num_experts: int = 8
    num_experts_per_tok: int = 2
    max_seq_len: int = 4096
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    router_aux_loss_coef: float = 0.01

    @classmethod
    def get_preset(cls, name: str):
        """Get preset configurations for Mixtral model sizes."""
        presets = {
            "mixtral-8x7b": cls(),
            # Small preset for the varying-shape sweep: hidden/intermediate 1024 keep
            # the expert matmuls routing to the dynamic mm kernel (K, N must be
            # 1024-multiples) while compiling cheaply. num_experts=4 with top_k=2
            # (< num_experts) so per-expert token counts vary step-to-step.
            "mixtral-tiny": cls(
                hidden_size=1024,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=128,
                intermediate_size=1024,
                num_experts=4,
                num_experts_per_tok=2,
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class MixtralAttention(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, rope_cache, is_causal=True):
        bs, seqlen, _ = x.shape
        xq = self.q_proj(x).view(bs, seqlen, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bs, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.o_proj(output)


class MixtralExpertFFN(nn.Module):
    """Single SwiGLU expert with fused gate_up projection (matches HuggingFace MixtralExperts)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class MixtralSparseMoE(nn.Module):
    """Top-k routed Mixture of Experts with load balancing loss.

    Reference HuggingFace implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py#MixtralSparseMoeBlock
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MixtralExpertFFN(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x):
        bs, seqlen, hidden = x.shape
        x_flat = x.view(-1, hidden)

        # Router: linear → softmax (float32 for numerical stability) → top-k
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Expert execution (for-loop, matching HuggingFace reference)
        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]

        for expert_idx, expert in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] > 0:
                expert_out = expert(x_flat[top_x])
                scaled = expert_out * top_k_weights[top_x, idx].unsqueeze(-1)
                out.index_add_(0, top_x, scaled.to(out.dtype))

        # Load balancing auxiliary loss
        tokens_per_expert = expert_mask.float().sum(dim=(1, 2))
        fraction_tokens = tokens_per_expert / (bs * seqlen)
        fraction_router = router_probs.mean(dim=0)
        aux_loss = self.aux_loss_coef * self.num_experts * (fraction_tokens * fraction_router).sum()

        return out.view(bs, seqlen, hidden), aux_loss.to(x.dtype)


class MixtralTransformerBlock(nn.Module):
    """Single Mixtral transformer block for benchmarking."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.attention = MixtralAttention(config)
        self.moe = MixtralSparseMoE(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(config.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

    def forward(self, x, is_causal=True):
        x = x + self.attention(self.attention_norm(x), self.rope_cache, is_causal=is_causal)
        normed = self.ffn_norm(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out
        # Return aux_loss for load balancing; must be added to training loss for router gradients
        return x, aux_loss
