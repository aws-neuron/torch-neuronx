# ruff: noqa: N806, E402, F841
"""
Custom decompositions for torch.compile neuron backend

This module contains custom decompositions that avoid operations
that cause issues in the torch-mlir to StableHLO conversion pipeline.
"""

import logging
import math
from collections.abc import Callable

import torch
from neuronxcc.nki.kernels import flash_fwd
from neuronxcc.nki.kernels.attention import FlashConfig

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.nki_hop import nki

logger = logging.getLogger(__name__)
wrapped_flash_fwd = nki.jit(flash_fwd)

# ops that are not supported by the backend (yet)
DECOMP_NAMES = {
    "aten._unsafe_index",
    "aten._unsafe_index_put",
    "aten._scaled_dot_product_flash_attention_for_cpu",
    "aten.split",
    "aten.split_with_sizes",
    "aten.as_strided",
    "aten.unbind",
    "aten._native_batch_norm_legit_no_training",
    "aten._native_batch_norm_legit",
    "aten._unsafe_view",
    "aten._softmax",
    "aten.flex_attention",
    "aten.topk",
    "aten.silu",
    "aten.index_copy",
    "aten.index_copy_",
}


def flex_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable | None = None,
    block_mask: torch.Tensor | None = None,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    chunk_size: int = 512,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-friendly decomposition of FlexAttention with chunked matmuls.
    Args:
        query: (B, H_q, L, E)
        key:   (B, H_kv, S, E)
        value: (B, H_kv, S, E)
        score_mod: Optional[Callable(scores, b, h, q_idx, k_idx)]
        block_mask: Optional[bool tensor] mask of shape (B, 1, L, S) or (1, 1, L, S)
        scale: optional scaling factor (default 1/sqrt(E))
        enable_gqa: enable grouped-query attention
        return_lse: if True, return (output, logsumexp)
        chunk_size: how many query tokens to process at once (controls memory/cache use)
    """
    B, H_q, L, E = query.shape
    _, H_kv, S, _ = key.shape
    device = query.device
    dtype = query.dtype
    scale = scale if scale is not None else (1.0 / (E**0.5))
    # Grouped Query Attention (expand K/V if needed)
    if enable_gqa and H_q != H_kv:
        assert H_q % H_kv == 0, "H_q must be multiple of H_kv for GQA"
        repeat_factor = H_q // H_kv
        key = key.unsqueeze(2).expand(B, H_kv, repeat_factor, S, E).reshape(B, H_q, S, E)
        value = value.unsqueeze(2).expand(B, H_kv, repeat_factor, S, E).reshape(B, H_q, S, E)
    outputs = []
    lse_chunks = [] if return_lse else None
    # Iterate over query chunks
    for q_start in range(0, L, chunk_size):
        q_end = min(q_start + chunk_size, L)
        q_chunk = query[:, :, q_start:q_end, :]  # (B, H, chunk, E)
        # (B, H, chunk, S)
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale
        # Optional score modification hook
        if score_mod is not None:
            b = torch.arange(B, device=device, dtype=torch.int32).view(B, 1, 1, 1)
            h = torch.arange(H_q, device=device, dtype=torch.int32).view(1, H_q, 1, 1)
            q_idx = torch.arange(q_start, q_end, device=device, dtype=torch.int32).view(
                1, 1, q_end - q_start, 1
            )
            k_idx = torch.arange(S, device=device, dtype=torch.int32).view(1, 1, 1, S)
            scores = score_mod(scores, b, h, q_idx, k_idx)
        # Optional block mask
        if block_mask is not None:
            # Expect mask shape (B, 1, L, S) or (1, 1, L, S)
            mask_slice = block_mask[..., q_start:q_end, :]
            scores = scores.masked_fill(~mask_slice, float("-inf"))
        # Numerically stable softmax
        max_score = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_score)
        denom = exp_scores.sum(dim=-1, keepdim=True)
        attn_weights = exp_scores / denom
        # Weighted sum over values
        output_chunk = torch.matmul(attn_weights, value)  # (B, H, chunk, E)
        outputs.append(output_chunk)
        if return_lse:
            lse_chunk = max_score.squeeze(-1) + torch.log(denom.squeeze(-1))
            lse_chunks.append(lse_chunk)
        # Free intermediate memory
        del q_chunk, scores, exp_scores, denom, attn_weights
    # Concatenate chunks
    output = torch.cat(outputs, dim=2)
    if return_lse:
        lse = torch.cat(lse_chunks, dim=2)
        return output, lse
    return output


def custom_as_strided_decomposition(input_tensor, size, stride, storage_offset=0):
    """
    Custom as_strided decomposition following torch-mlir pattern

    This implementation follows the exact decomposition pattern used in torch-mlir:
    https://github.com/llvm/torch-mlir/blob/main/lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp

    The pattern computes indices for each dimension and accumulates them to create
    the final indexing pattern, then uses advanced indexing to gather elements.

    Args:
        input_tensor: Input tensor to stride
        size: Target size tuple
        stride: Stride tuple
        storage_offset: Storage offset (default: 0)

    Returns:
        Tensor with the specified striding applied
    """
    # Get the device from the input tensor to ensure all intermediate tensors
    # are created on the same device
    device = input_tensor.device

    # Step 1: Flatten input tensor
    input_flat = torch.reshape(input_tensor, (-1,))

    # Step 2: Initialize index tensor (will accumulate indices for each dimension)
    # Start with zeros in the shape of the output, on the same device as input
    idx = torch.zeros(size, dtype=torch.long, device=device)

    # Step 3: For each dimension, compute its contribution to the final indices
    for dim, s in enumerate(size):
        # Create arange for this dimension on the same device
        arange = torch.arange(s, dtype=torch.long, device=device)

        # Create view shape: [1, 1, ..., -1, 1, 1, ...] where -1 is at position dim
        view_shape = []
        for i in range(len(size)):
            if i == dim:
                view_shape.append(-1)
            else:
                view_shape.append(1)

        # Reshape arange to broadcast properly
        arange = torch.reshape(arange, view_shape)

        # Add this dimension's contribution to the index
        # (skip dim 0 in the original pattern, but we'll handle all dims)
        idx = idx + arange * stride[dim]

    # Step 4: Flatten indices and add storage offset
    final_indices = torch.reshape(idx, (-1,)) + storage_offset

    # Step 5: Index the flattened input tensor using advanced indexing
    output = torch.index_select(input_flat, 0, final_indices)

    # Step 6: Reshape to desired output size
    return torch.reshape(output, size)


def scaled_dot_product_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Decomposition of scaled_dot_product_attention into basic ops.

    Args:
        query: (B, H, L, E) or (B, L, H, E) depending on layout
        key: (B, H, S, E) or (B, S, H, E)
        value: (B, H, S, E) or (B, S, H, E)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (ignored in decomposition)
        is_causal: Whether to apply causal masking
        scale: Optional scale factor (default: 1/sqrt(E))
        enable_gqa: Enable grouped-query attention

    Returns:
        Attention output tensor
    """
    # Get dimensions - assume (B, H, L, E) layout
    L = query.shape[-2]
    S = key.shape[-2]
    E = query.shape[-1]
    device = query.device
    dtype = query.dtype

    # Default scale
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    # Compute attention scores: (B, H, L, S)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        # Create causal mask: positions can only attend to earlier positions
        # Use scores.new_ones to derive tensor from existing tensor (avoids device issues)
        causal_mask = torch.triu(scores.new_ones(L, S, dtype=torch.bool), diagonal=S - L + 1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Apply attention mask if provided
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    # Softmax over key dimension
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, value)

    return output


def sdpa_fused_overrideable_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float | None = None,
):
    """
    Decomposition for _scaled_dot_product_fused_attention_overrideable.

    This op returns multiple outputs but we only need the attention output.
    The other outputs are for debugging/backward pass.
    """
    B, H, L, E = query.shape
    S = key.shape[-2]
    D = key.shape[-1]
    nki_attention_enabled = _getenv_bool("TORCH_NEURONX_ENABLE_NKI_SDPA", True)
    # Dynamically determine seq_tile_size based on sequence length
    # Get default from FlashConfig
    default_seq_tile_size = FlashConfig().seq_tile_size
    training = dropout_p > 0.0 or torch.is_grad_enabled()
    # Prefer default tile size when possible for optimal memory usage
    if S % default_seq_tile_size == 0:
        # Keep the default - this is the preferred case
        seq_tile_size = default_seq_tile_size
    else:
        # Only change if we must - prefer larger tiles up to default for better perf
        # We know this works due to our can_handle check (seq_len % 512 == 0)
        seq_tile_size = 1024 if S % 1024 == 0 else 512

    # Conditions to use NKI SDPA kernel
    can_use_nki_sdpa = L % 128 == 0 and S % 128 == 0 and D <= 128
    can_use_nki_sdpa = can_use_nki_sdpa and S % seq_tile_size == 0
    # Attn_bias is not yet supported
    can_use_nki_sdpa = can_use_nki_sdpa and attn_bias is None

    # Create config with custom seq_tile_size
    config = FlashConfig(training=training, seq_tile_size=seq_tile_size)
    if can_use_nki_sdpa and nki_attention_enabled:
        if dropout_p > 0.0:
            seed = torch.randint(0, 2**31 - 1, (1,), device=query.device, dtype=torch.int32)
        else:
            seed = None
        q = query.transpose(2, 3)
        k = key.transpose(2, 3)
        v = value  # default should_transpose_v is False

        output, logsumexp = wrapped_flash_fwd[query.shape[0], query.shape[1]](
            q,
            k,
            v,
            seed,
            logit_bias=attn_bias,
            softmax_scale=scale,
            use_causal_mask=is_causal,
            sliding_window=-1,
            mixed_precision=True,
            dropout_p=dropout_p,
            config=config,
        )

    else:
        # Compute the actual attention
        output = scaled_dot_product_attention_decomposition(
            query, key, value, attn_bias, dropout_p, is_causal, scale
        )

        # Create dummy outputs derived from existing tensors to avoid torch.tensor() calls
        # which can introduce detach_ operations on non-standard devices
        # Use query[0,0,0,0] * 0 to get a scalar on the same device without explicit tensor creation
        logsumexp = query.new_zeros(B, H, L)

    # Create scalar zeros derived from query tensor
    philox_seed = (query.flatten()[0] * 0).to(torch.int64)
    philox_offset = (query.flatten()[0] * 0).to(torch.int64)

    return (
        output,
        logsumexp,
        None,  # cum_seq_q
        None,  # cum_seq_k
        L,  # max_q
        S,  # max_k
        philox_seed,
        philox_offset,
        None,  # debug_mask
    )


# Fixed the torch.constant.int issue for embedding
def scalar_tensor_decomposition(
    value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    """
    Decomposition for scalar_tensor that avoids device argument issues.

    scalar_tensor creates a 0-d tensor from a scalar value. We decompose it to
    full() which is better supported in the torch-mlir StableHLO lowering.
    """
    # Use full with empty shape to create a scalar tensor
    return torch.full((), value, dtype=dtype, device=device)


def index_copy_decomposition(
    input: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """
    Decompose index_copy into functional scatter for AOT Autograd compatibility.

    index_copy copies values from source into input at positions specified by index along dim.
    """
    if dim < 0:
        dim = input.dim() + dim

    # Build target shape: 1s everywhere except at dim where we keep index size
    target_shape = [1] * source.dim()
    target_shape[dim] = index.numel()
    index_expanded = index.view(target_shape).expand(source.shape)
    return torch.scatter(input, dim, index_expanded, source)


# Registry of custom decompositions
CUSTOM_DECOMPOSITIONS = {
    torch.ops.aten.as_strided.default: custom_as_strided_decomposition,
    torch.ops.aten.scalar_tensor.default: scalar_tensor_decomposition,
    torch.ops.aten.index_copy.default: index_copy_decomposition,
}

# Register SDPA decompositions
CUSTOM_DECOMPOSITIONS[torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default] = (
    sdpa_fused_overrideable_decomposition
)


def get_custom_decomposition_table():
    """
    Get the custom decomposition table

    Returns:
        dict: Custom decomposition table for problematic operations
    """
    logger.debug(f"Providing {len(CUSTOM_DECOMPOSITIONS)} custom decompositions")
    for op_name in CUSTOM_DECOMPOSITIONS:
        logger.debug(f"  - {op_name}")

    return CUSTOM_DECOMPOSITIONS


def get_decomposition_table(decompose_all: bool = False) -> dict:
    """
    Get the complete decomposition table for the Neuron backend

    Args:
        decompose_all: If True, decompose all operations. If False,
            only decompose ops in DECOMP_NAMES.

    Returns:
        Decomposition table for the backend
    """
    from torch._decomp import decomposition_table

    # Start with our custom decompositions
    final_decomp_table = get_custom_decomposition_table()
    logger.debug(f"Added {len(final_decomp_table)} custom decompositions")

    if decompose_all:
        logger.debug("Decomposing all ops - merging with standard decomposition table")
        # Merge with all standard decompositions (custom ones take precedence)
        merged_table = decomposition_table.copy()
        merged_table.update(final_decomp_table)
        logger.debug(f"Final decomposition table has {len(merged_table)} entries (all ops)")
        return merged_table

    # iterate over the decomps and if key.op in decomp_names, add it to the decomp table
    for decomp_key, decomp_value in decomposition_table.items():
        # the str(decomp_key) may have the format: <op_name>.<default OR overload_name>
        # so try matching two things: either itself or with the last .<> part stripped
        if isinstance(decomp_key, torch._ops.OpOverload):
            op_name = "{}.{}".format(*decomp_key._schema.name.split("::"))
            if op_name in DECOMP_NAMES:
                if op_name not in final_decomp_table:
                    final_decomp_table[decomp_key] = decomp_value
                    logger.debug(f"Added standard decomposition for: {decomp_key}")
                else:
                    logger.debug(f"Using custom decomposition for: {decomp_key}")
        else:
            logger.debug(
                f"Found torch op {decomp_key} which is not a torch._ops.OpOverload instance - "
                + "not adding to decomposition table"
            )

    logger.debug(f"Final decomposition table has {len(final_decomp_table)} entries (selective)")
    return final_decomp_table
