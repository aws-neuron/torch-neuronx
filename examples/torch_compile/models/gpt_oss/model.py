# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from .args import GptOssModelArgs
from .attention import build_attention
from .moe import GptOssMoE
from .rope_patched import (
    apply_rotary_emb_patched,
    precompute_rope_cache_patched,
    repeat_kv_patched,
)

AttentionMasksType = Optional[dict[str, torch.Tensor]]  # noqa: UP007


class Attention(nn.Module):
    """
    Multi-head attention module with support for FlexAttention.
    """

    def __init__(self, model_args: GptOssModelArgs, use_sliding_window: bool = False):
        super().__init__()
        self.head_dim = model_args.head_dim
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads

        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            model_args.dim,
            model_args.n_heads * model_args.head_dim,
            bias=True,
        )
        self.wk = nn.Linear(
            model_args.dim,
            model_args.n_kv_heads * model_args.head_dim,
            bias=True,
        )
        self.wv = nn.Linear(
            model_args.dim,
            model_args.n_kv_heads * model_args.head_dim,
            bias=True,
        )
        self.wo = nn.Linear(
            model_args.n_heads * model_args.head_dim,
            model_args.dim,
            bias=True,
        )

        # Build attention module based on configuration
        sliding_window = model_args.sliding_window_size if use_sliding_window else None
        self.attention = build_attention(
            use_flex_attn=model_args.use_flex_attn,
            attn_mask_type=model_args.attn_mask_type,
            sliding_window_size=sliding_window,
        )

    def init_weights(self, init_std: float):
        linear_list = [
            self.wq,
            self.wk,
            self.wv,
        ]

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            nn.init.trunc_normal_(linear.bias, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.wo.bias, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Simplified forward pass using ScaledDotProductAttention.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies for rope embedding.
        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb_patched(q, k, rope_cache)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv_patched(k, self.n_rep)
        values = repeat_kv_patched(v, self.n_rep)

        xq = q.transpose(1, 2).contiguous()  # (B, T, H, D) -> (B, H, T, D)
        xk = keys.transpose(1, 2).contiguous()
        xv = values.transpose(1, 2).contiguous()

        # Use the simplified attention module
        output = self.attention(xq, xk, xv)

        output = output.transpose(1, 2).contiguous()  # (B, H, T, D) -> (B, T, H, D)

        # Reshape and project output
        output = output.reshape(bsz, seqlen, -1).contiguous()  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)  # (bsz, seqlen, dim)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: GptOssModelArgs):
        super().__init__()
        self.use_sliding_attention = layer_id % 2 == 0
        # Pass sliding window flag to attention module
        self.attention = Attention(model_args, use_sliding_window=self.use_sliding_attention)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Add MoE
        self.moe = GptOssMoE(
            model_args.moe_args, dim=model_args.dim, hidden_dim=model_args.moe_inter_dim
        )
        self.moe_enabled = True  # for composability with load balancing

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
        """
        Forward pass for the Transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # Attention with proper masking (causal or causal+sliding window)
        x = x + self.attention(self.attention_norm(x), rope_cache)

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.moe.init_weights(self.weight_init_std, buffer_device)


class GptOssModel(nn.Module):
    """
    GPT-OSS Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: GptOssModelArgs):
        super().__init__()
        self.model_args = model_args
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("rope_cache", self._precompute_rope_cache(), persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            bias=False,
        )
        self.model_args = model_args
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache_patched(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ):
        """
        Forward pass for the Transformer model.
        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            attention_masks (AttentionMasksType): a dict of BlockMasks.
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        h = self.tok_embeddings(tokens)

        for layer in self.layers.values():
            h = layer(h, self.rope_cache)
        h = self.norm(h)
        output = self.output(h)
        return output
