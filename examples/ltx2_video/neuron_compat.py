"""Neuron compatibility shims for LTX-2 native PyTorch on Beta 3.

Three shims, ALL needed for correct output (the AWS Neuron team's
contrib `aws-neuron/neuronx-distributed-inference/contrib/models/
ltx2-video-audio` documents these as the production fixes; see
`modeling_ltx2.py` upstream):

1. `install_bmm_sdpa()` — replaces `F.scaled_dot_product_attention` with
   explicit BMM + softmax + add-mask. Stock SDPA on Neuron's compiled
   bf16 lazy backend miscomputes for LTX-2 (damped values + ±11
   outliers), producing washed-out video. AWS-team docstring:
   "Replaces SDPA with explicit BMM operations for Neuron XLA
   compatibility."

2. `RankTensor` — a tiny `nn.Module` that holds
   `torch.arange(world_size)` as a parameter, so per-rank slicing
   uses a TRACED tensor value (not a Python int that gets baked as
   constant 0 during XLA tracing — which would make all ranks apply
   the same RoPE shard). This is the SPMDRank pattern from the AWS
   contrib, simplified for non-NxD use.

3. `to_additive_mask()` — converts a {0,1} bool/int attention mask into
   an additive `-10000.0` bias (with bf16-safe value). Critical: must
   be `-10000.0`, NOT `0.0`. AWS-team docstring on
   `_make_full_attn_mask`: "Using all-zeros causes XLA to constant-fold
   the mask, dropping it from the compiled graph."

Usage from your run script:

    from neuron_compat import install_bmm_sdpa, RankTensor, to_additive_mask

    install_bmm_sdpa()                       # call ONCE before any forward
    rank_tensor = RankTensor(world_size, rank)  # for per-rank RoPE slicing
    mask = to_additive_mask(mask_bool, dtype=torch.bfloat16)

License: Apache-2.0 (matches the AWS contrib license).
"""
from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Module-level guard so install is idempotent ─────────────────────────────
_sdpa_replaced = False
_sdpa_original = None


def install_bmm_sdpa() -> None:
    """Replace torch.nn.functional.scaled_dot_product_attention with BMM.

    Idempotent. CPU calls fall through to the original SDPA so the text
    encoder + VAE on CPU still get the optimized SDPA path.

    Implementation lifted (BMM fallback only) from
    aws-neuron/neuronx-distributed-inference/contrib/models/
    ltx2-video-audio/src/modeling_ltx2.py::replace_sdpa_with_bmm
    """
    global _sdpa_replaced, _sdpa_original
    if _sdpa_replaced:
        return
    _sdpa_original = torch.nn.functional.scaled_dot_product_attention

    def _neuron_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False):
        # CPU fallback (text encoder + VAE preprocessing on CPU)
        if query.device.type == "cpu":
            return _sdpa_original(
                query, key, value,
                attn_mask=attn_mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale,
            )

        d = query.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)

        orig_shape = None
        if len(query.shape) == 4:
            orig_shape = query.shape
            b, h, sq, d_head = query.shape
            query = query.reshape(b * h, sq, d_head)
            key = key.reshape(b * h, -1, d_head)
            value = value.reshape(b * h, -1, d_head)
            if attn_mask is not None:
                if attn_mask.ndim == 4:
                    attn_mask = attn_mask.reshape(
                        b * h, attn_mask.shape[-2], attn_mask.shape[-1]
                    )
                elif attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                elif attn_mask.ndim == 3 and attn_mask.shape[0] == orig_shape[0]:
                    attn_mask = (
                        attn_mask.unsqueeze(1)
                        .expand(orig_shape[0], orig_shape[1], -1, -1)
                        .reshape(
                            orig_shape[0] * orig_shape[1],
                            attn_mask.shape[-2],
                            attn_mask.shape[-1],
                        )
                    )

        scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = scores.softmax(dim=-1)
        out = torch.bmm(probs, value)

        if orig_shape is not None:
            out = out.reshape(orig_shape[0], orig_shape[1], -1, orig_shape[3])
        return out

    torch.nn.functional.scaled_dot_product_attention = _neuron_sdpa
    _sdpa_replaced = True
    logger.info("BMM-SDPA replacement installed (Neuron XLA compatibility)")


# ── RankTensor (lightweight SPMDRank-style replacement) ─────────────────────
class RankTensor(nn.Module):
    """Holds a per-rank index as a TRACED tensor, not a Python int.

    Why: in XLA SPMD tracing, a Python `rank * h_local` baked into a
    `tensor[start:end]` op compiles to `tensor[0:h_local]` for ALL ranks
    (the trace runs once on rank 0). The result is every rank applies
    the SAME RoPE shard — silently wrong, no error.

    This module exposes `.value()` returning a 0-d int32 tensor (the
    rank's index), populated from `torch.arange(world_size)[rank]`. Each
    rank's traced graph references its own arange element, so slicing
    ops parameterized by `.value() * h_local` produce the per-rank
    correct slice.

    The AWS Neuron contrib uses NxD's `SPMDRank`; this class matches its
    API surface for our usage (a single int) without the NxD dependency.
    """

    def __init__(self, world_size: int, rank: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        # arange so each rank's compiled graph sees its OWN element
        ranks = torch.arange(world_size, dtype=torch.int32)
        self.register_buffer("rank_arange", ranks, persistent=False)
        self._rank_int = rank  # plain Python value for non-traced contexts

    def value(self) -> torch.Tensor:
        """Return a 0-d int32 tensor for this rank.

        Inside a traced forward, prefer this over `self.rank` so the
        slice/index op references the traced `arange[rank]` value.
        """
        return self.rank_arange[self._rank_int]

    def slice_dim(
        self,
        tensor: torch.Tensor,
        dim: int,
        full_size: int,
    ) -> torch.Tensor:
        """Slice `tensor` along `dim` to this rank's chunk of `full_size`.

        Equivalent to `tensor.select(dim, rank_lo:rank_hi)` where
        `rank_lo = rank * (full_size // world_size)` etc., but uses the
        traced rank tensor so XLA SPMD generates per-rank slices.
        """
        chunk = full_size // self.world_size
        rank_t = self.value()
        start = rank_t * chunk
        # narrow takes a Python int for the size, OK here (chunk is constant)
        return torch.narrow(tensor, dim, start, chunk)


# ── Additive mask conversion ────────────────────────────────────────────────
def to_additive_mask(
    mask: torch.Tensor | None,
    dtype: torch.dtype = torch.bfloat16,
    masked_value: float = -10000.0,
) -> torch.Tensor | None:
    """Convert a {0,1} attention mask to an additive bias mask.

    Conventions:
      Input mask: 1 = ATTEND, 0 = MASK OUT (the diffusers convention).
      Output    : 0.0 for ATTEND, `masked_value` for MASK OUT.

    Why -10000.0 and not -inf:
      bf16 saturates -inf to -1e38; subsequent softmax produces NaN on
      Neuron's compiled lazy backend. -10000.0 is the standard
      Hugging-Face / NxDI bias value; produces ~0 after softmax.

    Why not 0.0 for ATTEND:
      An all-zeros mask gets constant-folded by XLA and DROPPED from the
      compiled graph (AWS team docstring). For a true "attend
      everywhere" prompt it doesn't matter functionally, but the
      compile-time elision then bites you when a real padded prompt
      arrives later. Always emit the additive form.
    """
    if mask is None:
        return None
    return ((1 - mask.to(dtype)) * masked_value).to(dtype)


# ── Diagnostic: print which fixes are active ─────────────────────────────────
def print_active_fixes(rank: int = 0) -> None:
    if rank != 0:
        return
    print("[neuron_compat] active fixes:", flush=True)
    print(f"  - BMM-SDPA installed: {_sdpa_replaced}", flush=True)
    print(f"  - RankTensor: import-only (instantiate per-model)", flush=True)
    print(f"  - to_additive_mask: import-only (call per-mask)", flush=True)
