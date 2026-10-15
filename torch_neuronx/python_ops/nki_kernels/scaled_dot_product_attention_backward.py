"""NKI kernel wrapper for scaled dot-product attention backward operation"""

import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import _compute_attention_backward_nki
from torch_neuronx.nki_hop import nki_op


@nki_op("nki_kernels::scaled_dot_product_attention_backward_kernel", mutates_args={})
def scaled_dot_product_attention_backward_kernel(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention gradients with the NKI kernel, transposing ATen layouts.

    Args:
        grad_out (torch.Tensor): Output gradient with shape ``(B, Hq, L, E)``.
        q (torch.Tensor): Query tensor with shape ``(B, Hq, L, E)``.
        k (torch.Tensor): Key tensor with shape ``(B, Hkv, S, E)``.
        v (torch.Tensor): Value tensor with shape ``(B, Hkv, S, E)``.
        out (torch.Tensor): Forward output with shape ``(B, Hq, L, E)``.
        lse (torch.Tensor): Log-sum-exp in ``(B, Hq, L)`` or ``(B, Hq, 128, L // 128)`` form.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to ``False``.
        dropout_p (float, optional): Ignored dropout probability. Defaults to ``0.0``.
        scale (float | None, optional): Softmax scale. Defaults to ``None`` (derive from ``E``).
        seed (int | None, optional): Ignored dropout seed. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients for ``q``, ``k``, and ``v``.

    Raises:
        ValueError: If a three-dimensional ``lse`` has ``L`` not divisible by 128.
    """
    return _compute_attention_backward_nki(
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        is_causal,
        scale,
    )
