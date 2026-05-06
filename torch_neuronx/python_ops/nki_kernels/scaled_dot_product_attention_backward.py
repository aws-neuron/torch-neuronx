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
    """
    Wrapper for the NKI flash attention backward kernel.

    Args:
        grad_out: Gradient of output (batch, heads, seq_len, embed_dim)
        q: Query tensor in ATen format (batch, heads, seq_len, embed_dim)
        k: Key tensor in ATen format (batch, heads, seq_len, embed_dim)
        v: Value tensor in ATen format (batch, heads, seq_len, embed_dim)
        out: Forward output (batch, heads, seq_len, embed_dim)
        lse: Log-sum-exp from forward (batch, heads, B_P_SIZE, n_tiles)
        is_causal: Whether causal masking was used
        dropout_p: Dropout probability
        scale: Softmax scale factor
        seed: Random seed for dropout

    Note:
        - Inputs are in ATen format; transpose to NKI format is done by
          _compute_attention_backward_nki
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
