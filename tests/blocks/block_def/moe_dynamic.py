"""Mixtral-8x7B MoE block for the dynamic-kernel benchmark.

Reuses Mixtral's GQA attention + config. The MoE FFN uses realistic top-k routing
with arbitrary per-expert token counts — the dynamic-NKI adapters pad each count up
to a 128-multiple internally, so no floor/capacity is needed here. The expert loop
(gather -> SwiGLU -> scatter) is the op pattern the dynamic kernels intercept.
Single device; no sharding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from block_def.mixtral_moe import MixtralAttention, MixtralConfig
from block_def.rms_norm import RMSNorm
from block_def.rope import precompute_rope_cache


class ExpertFFN(nn.Module):
    """Single SwiGLU expert with fused gate_up projection."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_w = nn.Parameter(torch.empty(hidden_size, 2 * intermediate_size))
        self.down_w = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        nn.init.normal_(self.gate_up_w, std=0.02)
        nn.init.normal_(self.down_w, std=0.02)

    def forward(self, x):
        gate, up = torch.mm(x, self.gate_up_w).chunk(2, dim=-1)
        return torch.mm(F.silu(gate) * up, self.down_w)


class SparseMoE(nn.Module):
    """Top-k routed MoE with per-expert token counts."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                ExpertFFN(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def forward(self, x):
        bs, seqlen, hidden = x.shape
        x_flat = x.reshape(-1, hidden)

        router_probs = F.softmax(self.gate(x_flat).float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx, expert in enumerate(self.experts):
            sel = torch.where(expert_mask[expert_idx].sum(dim=0) > 0)[0]
            if sel.shape[0] == 0:
                continue
            wsum = (top_k_weights * (top_k_indices == expert_idx)).sum(dim=-1, keepdim=True)
            w_sel = wsum[sel]  # [n, 1]
            expert_out = expert(x_flat[sel])  # gather -> SwiGLU
            scaled = expert_out * w_sel.expand_as(expert_out).to(out.dtype)
            scatter_idx = sel.to(torch.int32).unsqueeze(1).expand_as(scaled)
            out.scatter_add_(0, scatter_idx, scaled)
        return out.reshape(bs, seqlen, hidden)


class MoEDynamicBlock(nn.Module):
    """Mixtral attention + top-k routed MoE FFN, RMSNorm pre-norms, cached RoPE."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.attention = MixtralAttention(config)
        self.moe = SparseMoE(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.register_buffer(
            "rope_cache",
            precompute_rope_cache(config.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

    def forward(self, x, is_causal=True):
        x = x + self.attention(self.attention_norm(x), self.rope_cache, is_causal=is_causal)
        x = x + self.moe(self.ffn_norm(x))
        return x
