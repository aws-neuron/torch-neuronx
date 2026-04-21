"""Neuron-safe rotary position embedding (RoPE) utilities.

The standard RoPE implementation used by HuggingFace diffusers and many
transformer libraries uses a pattern that produces non-finite values on
Neuron hardware::

    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

The combination of ``torch.empty_like`` (uninitialized memory) with
stride-2 slice assignment triggers a non-contiguous write path whose
interaction with the Neuron compiler produces inf/NaN in the output.
CPU execution of the identical formula produces correct results.

This module provides :func:`apply_rotary_emb` which computes the same
mathematical result using only contiguous operations (``torch.stack`` +
``flatten``) that the Neuron backend handles correctly.

See Also
--------
- ``docs/rope_neuron_fix.md`` for the full analysis.
- ``examples/rope_neuron_minimal_repro.py`` for a standalone reproducer.
"""

from __future__ import annotations

import torch


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding using Neuron-safe contiguous operations.

    Mathematically equivalent to the diffusers ``WanAttnProcessor`` RoPE::

        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos

    but avoids ``empty_like`` + stride-2 slice writes that produce inf/NaN
    on the Neuron backend.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(..., D)`` where ``D`` is even.  Typically bf16 on Neuron.
    freqs_cos : torch.Tensor
        Cosine frequencies, shape broadcastable to ``(..., D)``.
    freqs_sin : torch.Tensor
        Sine frequencies, shape broadcastable to ``(..., D)``.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as *hidden_states*.
    """
    x = hidden_states.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(-1)

    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    o_even = x1 * cos - x2 * sin
    o_odd = x1 * sin + x2 * cos

    return torch.stack([o_even, o_odd], dim=-1).flatten(-2).type_as(hidden_states)
