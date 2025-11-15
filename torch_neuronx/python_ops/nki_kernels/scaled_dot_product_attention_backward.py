"""NKI kernel wrapper for scaled dot-product attention backward operation"""

import math

import neuronxcc.nki.language as nl
from neuronxcc.nki.kernels.attention import flash_attn_bwd
from torch_neuronx.utils import suppress_specific_warnings


def scaled_dot_product_attention_backward_kernel(
    grad_out,
    q,
    k,
    v,
    out,
    lse,
    grad_q,
    grad_k,
    grad_v,
    is_causal=False,
    dropout_p=0.0,
    scale=None,
    seed=None,
):
    """
    Wrapper for the NKI flash attention backward kernel.

    Args:
        grad_out: Gradient of output (batch, heads, seq_len, embed_dim)
        q: Query tensor in NKI format (batch, heads, embed_dim, seq_len)
        k: Key tensor in NKI format (batch, heads, embed_dim, seq_len)
        v: Value tensor in NKI format (batch, heads, embed_dim, seq_len)
        out: Forward output (batch, heads, seq_len, embed_dim)
        lse: Log-sum-exp from forward (batch, heads, B_P_SIZE, n_tiles)
        grad_q: Output gradient for Q (batch, heads, embed_dim, seq_len)
        grad_k: Output gradient for K (batch, heads, embed_dim, seq_len)
        grad_v: Output gradient for V (batch, heads, embed_dim, seq_len)
        is_causal: Whether causal masking was used
        dropout_p: Dropout probability
        scale: Softmax scale factor
        seed: Random seed for dropout
    """
    # Import here to avoid circular import
    from torch_neuronx import TorchNeuronNKIKernel
    from torch_neuronx.utils import get_logical_neuron_cores

    # LNC sharding factor for NKI kernel based on available neuron cores
    logical_neuron_cores = int(get_logical_neuron_cores())

    # Get dimensions
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    embed_dim = q.shape[2]

    # Calculate softmax scale if not provided
    if scale is None:
        scale = 1.0 / math.sqrt(embed_dim)

    # Need to transpose grad_out and out to NKI format
    # From (batch, heads, seq_len, embed_dim) to (batch, heads, embed_dim, seq_len)
    grad_out_nki = grad_out.transpose(-2, -1)
    out_nki = out.transpose(-2, -1)

    # V needs to be transposed for backward kernel
    # From (batch, heads, seq_len, embed_dim) to (batch, heads, embed_dim, seq_len)
    v_nki = v.transpose(-2, -1)

    # Create the kernel grid with LNC sharding on head dimension
    if logical_neuron_cores > 1 and num_heads % logical_neuron_cores == 0:
        grid = (batch_size, nl.nc(logical_neuron_cores) * (num_heads // logical_neuron_cores))
    else:
        grid = (batch_size, num_heads)

    traced_kernel = TorchNeuronNKIKernel(
        func=getattr(flash_attn_bwd, "func", flash_attn_bwd),
        grid=grid,
        kernel_return=True,
        return_tensor_overrides=(grad_q, grad_k, grad_v),
    )

    # Prepare seed tensor - needs to be shape (1,) not (1, 1)
    seed_1d = seed.view(-1)[0:1] if seed is not None and seed.numel() > 0 else None
    with suppress_specific_warnings(["Block dimension is deprecated", "shadowing tile"]):
        dq_ret, dk_ret, dv_ret = traced_kernel(
            q,
            k,
            v_nki,
            out_nki,
            grad_out_nki,
            lse,
            seed_1d,
            use_causal_mask=is_causal,
            mixed_precision=True,
            dropout_p=dropout_p,
            softmax_scale=scale,
            logit_bias_ref=None,
        )
    return grad_q, grad_k, grad_v
