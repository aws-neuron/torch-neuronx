# ruff: noqa: N812

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from TorchTitan for use with Neuron torch.compile backend

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled dot product attention using PyTorch's SDPA.
    Simplified version for Neuron backend - only supports causal masking.
    """

    def __init__(self, attn_mask_type: str = "causal") -> None:
        super().__init__()
        if attn_mask_type != "causal":
            raise ValueError("Neuron backend currently only supports causal mask.")
        self.backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        with sdpa_kernel(self.backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def build_attention(use_flex_attn: bool = False, attn_mask_type: str = "causal"):
    """
    Build attention module.
    For Neuron backend, we use standard SDPA (FlexAttention not supported yet).
    """
    if use_flex_attn:
        raise NotImplementedError("FlexAttention not yet supported with Neuron backend")
    return ScaledDotProductAttention(attn_mask_type)
