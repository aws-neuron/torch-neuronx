# ruff: noqa: N812

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchTitan for use with Neuron torch.compile backend

from typing import Optional

import torch
import torch.nn.attention.flex_attention as flex_attn_module
from torch.nn.attention.flex_attention import flex_attention

_original_validate_device = flex_attn_module._validate_device


def _patched_validate_device(query, key, value):
    if query.device.type == "neuron":
        return  # Skip validation for neuron, let it through for tracing
    return _original_validate_device(query, key, value)


flex_attn_module._validate_device = _patched_validate_device


class FlexAttention(torch.nn.Module):
    """
    FlexAttention implementation using FX pass for Neuron backend.

    Supports causal masking and sliding window attention patterns.
    """

    def __init__(
        self, attn_mask_type: str = "causal", sliding_window_size: int | None = None
    ) -> None:
        super().__init__()
        self.attn_mask_type = attn_mask_type
        self.sliding_window_size = sliding_window_size

        # Create the score modification function based on mask type
        if attn_mask_type == "causal":
            if sliding_window_size is not None:
                # Causal + sliding window
                self.score_mod = self._causal_sliding_window_score_mod
            else:
                # Pure causal
                self.score_mod = self._causal_score_mod
        else:
            raise ValueError(f"Unsupported attention mask type: {attn_mask_type}")

    def _causal_score_mod(self, score, batch, head, q_idx, k_idx):
        """Causal masking: only attend to current and previous positions"""
        mask = q_idx < k_idx  # positions that should be masked (future positions)
        return score.masked_fill(mask, float("-inf"))

    def _causal_sliding_window_score_mod(self, score, batch, head, q_idx, k_idx):
        """Causal + sliding window: attend to previous positions within window"""
        causal_mask = q_idx < k_idx  # future positions
        window_mask = (q_idx - k_idx) > self.sliding_window_size  # outside window
        combined_mask = causal_mask | window_mask
        return score.masked_fill(combined_mask, float("-inf"))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using flex_attention.

        Args:
            q: Query tensor (B, H, L, E)
            k: Key tensor (B, H, S, E)
            v: Value tensor (B, H, S, E_v)
            scale: Optional scale factor

        Returns:
            Attention output (B, H, L, E_v)
        """

        return flex_attention(
            q,
            k,
            v,
            score_mod=self.score_mod,
            scale=scale,
            enable_gqa=False,
            return_lse=False,
        )


def build_attention(
    attn_mask_type: str = "causal", sliding_window_size: int | None = None, **kwargs
):
    """
    Build attention module using FlexAttention.

    Args:
        attn_mask_type: Type of attention mask ("causal")
        sliding_window_size: Optional sliding window size for local attention
        **kwargs: Ignored for backward compatibility (e.g., use_flex_attn)

    Returns:
        FlexAttention module instance
    """
    return FlexAttention(attn_mask_type, sliding_window_size)
