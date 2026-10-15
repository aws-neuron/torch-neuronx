"""NKI kernel wrapper for scaled dot-product attention operation"""

import torch

from torch_neuronx.neuron_dynamo_backend.decompositions import _compute_attention_nki
from torch_neuronx.nki_hop import nki_op


@nki_op("nki_kernels::scaled_dot_product_attention_kernel", mutates_args={})
def _scaled_dot_product_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    dropout_p: float,
    scale: float | None,
    training: bool,
    seed: int | None,
    lse: torch.Tensor | None,
) -> torch.Tensor:
    """
    Wrapper for the NKI flash attention kernel for scaled_dot_product_attention.

    Args:
        q: Query tensor in ATen format (batch, heads, seq_len, embed_dim)
        k: Key tensor in ATen format (batch, kv_heads, seq_len, embed_dim)
        v: Value tensor in ATen format (batch, kv_heads, seq_len, embed_dim)
        is_causal: Whether to apply causal masking
        dropout_p: Dropout probability
        scale: Softmax scale factor
        training: Whether in training mode
        seed: Random seed for dropout (if dropout_p > 0)
        lse: Log-sum-exp buffer for backward pass (if training)
    """
    if training:
        attn_output, lse_ret = _compute_attention_nki(q, k, v, is_causal, scale, training)
        if lse is not None and lse_ret is not lse:
            lse.copy_(lse_ret)
    else:
        attn_output = _compute_attention_nki(q, k, v, is_causal, scale, training)
    return attn_output


def scaled_dot_product_attention_kernel(
    q,
    k,
    v,
    is_causal=False,
    dropout_p=0.0,
    scale=None,
    training=False,
    seed=None,
    lse=None,
):
    """Run scaled dot-product attention through the NKI op wrapper.

    Args:
        q (torch.Tensor): Query tensor in ``(batch, heads, sequence, embedding)`` format.
        k (torch.Tensor): Key tensor in ``(batch, kv_heads, sequence, embedding)`` format.
        v (torch.Tensor): Value tensor in ``(batch, kv_heads, sequence, embedding)`` format.
        is_causal (bool, optional): Whether to apply causal masking. Defaults to ``False``.
        dropout_p (float, optional): Compatibility dropout probability. Defaults to ``0.0``.
        scale (float | None, optional): Softmax scaling factor. Defaults to ``None``.
        training (bool, optional): Whether to request training outputs. Defaults to ``False``.
        seed (int | None, optional): Compatibility dropout seed. Defaults to ``None``.
        lse (torch.Tensor | None, optional): Log-sum-exp destination for training.
            Defaults to ``None``.

    Returns:
        torch.Tensor: Attention output.

    Note:
        The current implementation does not consume ``dropout_p`` or ``seed``. In
        training mode it copies the returned log-sum-exp values into ``lse`` when needed.
    """
    return _scaled_dot_product_attention_kernel(
        q,
        k,
        v,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
        training=training,
        seed=seed,
        lse=lse,
    )
