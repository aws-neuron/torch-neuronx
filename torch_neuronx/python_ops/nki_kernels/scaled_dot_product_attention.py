"""NKI kernel wrapper for scaled dot-product attention operation"""

import math

import neuronxcc.nki.language as nl
import torch
from neuronxcc.nki.kernels.attention import FlashConfig, flash_fwd

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import suppress_specific_warnings

wrapped_flash_fwd = wrap_nki(flash_fwd)


@nki_op("nki_kernels::scaled_dot_product_attention_kernel", mutates_args={})
def _scaled_dot_product_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_output: torch.Tensor,
    is_causal: bool,
    dropout_p: float,
    scale: float | None,
    is_gqa: bool,
    training: bool,
    seed: int | None,
    lse: torch.Tensor | None,
    should_transpose_v: bool,
) -> torch.Tensor:
    """
    Wrapper for the NKI flash attention kernel for scaled_dot_product_attention.

    This uses the same underlying NKI kernel as _native_multi_head_attention's
    middle operation, but with different input handling for SDPA.

    Args:
        q: Query tensor in NKI format (batch, heads, embed_dim, seq_len)
        k: Key tensor in NKI format (batch, heads, embed_dim, seq_len)
        v: Value tensor in ATen format (batch, heads, seq_len, embed_dim)
        attn_output: Output tensor (batch, heads, seq_len, embed_dim)
        is_causal: Whether to apply causal masking
        dropout_p: Dropout probability
        scale: Softmax scale factor
        is_gqa: Whether this is grouped query attention
        training: Whether in training mode
        seed: Random seed for dropout (if dropout_p > 0)
        lse: Log-sum-exp buffer for backward pass (if training)

    Note:
        - V is kept in ATen format to avoid transpose (should_transpose_v=False)
        - For GQA, the grid uses kv_heads, not query_heads
    """
    # Import here to avoid circular import
    from torch_neuronx.utils import get_logical_neuron_cores

    # LNC sharding factor for NKI kernel based on available neuron cores
    logical_neuron_cores = int(get_logical_neuron_cores())

    # Get dimensions from Q (which has all query heads)
    batch_size = q.shape[0]
    q_heads = q.shape[1]
    embed_dim = q.shape[2]
    seq_len = q.shape[3]

    # Get KV heads from K tensor
    kv_heads = k.shape[1]

    # Validate GQA setup
    if is_gqa:
        assert (
            q_heads % kv_heads == 0
        ), f"Query heads ({q_heads}) must be divisible by KV heads ({kv_heads}) for GQA"

    if scale is None:
        scale = 1 / math.sqrt(embed_dim)

    # Determine seq_tile_size based on sequence length
    # Get default from FlashConfig
    default_seq_tile_size = FlashConfig().seq_tile_size

    # Prefer default tile size when possible for optimal memory usage
    if seq_len % default_seq_tile_size == 0:
        seq_tile_size = default_seq_tile_size
    else:
        # Only change if we must - prefer larger tiles for better performance
        # We know seq_len % 512 == 0 from can_handle check
        seq_tile_size = 1024 if seq_len % 1024 == 0 else 512

    # Create config based on training mode
    # Important: should_transpose_v=False because we're passing V in
    # (batch, heads, seq_len, embed_dim) format to avoid transpose
    config = FlashConfig(
        training=training,  # Use the passed training flag
        should_transpose_v=should_transpose_v,  # V is already in correct format
        seq_tile_size=seq_tile_size,
    )

    # Create the kernel with grid
    # For GQA, use kv_heads in grid, not q_heads!
    # The kernel internally handles the q_heads/kv_heads ratio
    grid_heads = kv_heads if is_gqa else q_heads

    # Shard on head dimension to use available logical neuron cores
    if logical_neuron_cores > 1 and grid_heads % logical_neuron_cores == 0:
        grid = (batch_size, nl.nc(logical_neuron_cores) * (grid_heads // logical_neuron_cores))
    else:
        grid = (batch_size, grid_heads)

    suppressed_warnings = ["Block dimension is deprecated", "shadowing tile"]
    if training:
        with suppress_specific_warnings(suppressed_warnings):
            out_ret, lse_ret = wrapped_flash_fwd[grid](
                q,
                k,
                v,
                seed=seed,
                use_causal_mask=is_causal,
                mixed_precision=True,
                softmax_scale=scale,
                dropout_p=dropout_p,
                config=config,
            )
        if out_ret is not attn_output:
            attn_output.copy_(out_ret)
        if lse is not None and lse_ret is not lse:
            lse.copy_(lse_ret)
    else:
        with suppress_specific_warnings(suppressed_warnings):
            out_ret = wrapped_flash_fwd[grid](
                q,
                k,
                v,
                seed,
                use_causal_mask=is_causal,
                mixed_precision=True,
                softmax_scale=scale,
                dropout_p=dropout_p,
                config=config,
            )
        if out_ret is not attn_output:
            attn_output.copy_(out_ret)

    # TODO: the attn_output needs to be initialized with this layout, since
    # pytorch expects it to be in this layout:
    # https://github.com/pytorch/pytorch/blob/main/torch/_meta_registrations.py#L5761
    # For single chip, this tranpose operations are not adding a perf hit on the
    # block test,so for now we are good, but need to fix the nki kernel to
    # produce output in this layout
    attn_output = attn_output.transpose(1, 2).contiguous().transpose(1, 2)

    return attn_output


def scaled_dot_product_attention_kernel(
    q,
    k,
    v,
    attn_output,
    is_causal=False,
    dropout_p=0.0,
    scale=None,
    is_gqa=False,
    training=False,
    seed=None,
    lse=None,
    should_transpose_v=False,
):
    return _scaled_dot_product_attention_kernel(
        q,
        k,
        v,
        attn_output,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
        is_gqa=is_gqa,
        training=training,
        seed=seed,
        lse=lse,
        should_transpose_v=should_transpose_v,
    )
