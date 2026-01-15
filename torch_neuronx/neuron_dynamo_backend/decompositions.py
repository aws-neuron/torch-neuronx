# ruff: noqa: N806, E402, F841
"""
Custom decompositions for torch.compile neuron backend

This module contains custom decompositions that avoid operations
that cause issues in the torch-mlir to StableHLO conversion pipeline.

See https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
for existing decompositions.
"""

import logging
import math
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch
import torch._decomp as decomp
from neuronxcc.nki.kernels import flash_attn_bwd, flash_fwd
from neuronxcc.nki.kernels.attention import FlashConfig
from torch._decomp import get_decompositions

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.nki_hop import wrap_nki

_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten
from torch_neuronx.utils import get_logical_neuron_cores

logger = logging.getLogger(__name__)
wrapped_flash_fwd = wrap_nki(flash_fwd)
wrapped_flash_bwd = wrap_nki(flash_attn_bwd)

# Build the base decomposition table from standard PyTorch decompositions
# for ops that are not supported by the backend (yet)
neuron_decompositions = get_decompositions(
    [
        aten._unsafe_index,
        aten._unsafe_index_put,
        aten.split,
        aten.split_with_sizes,
        aten.unbind,
        aten.replication_pad2d,
        aten.reflection_pad2d,
        aten.embedding,
        aten.embedding_dense_backward,
        aten.native_layer_norm_backward,
        aten.nll_loss_backward,
        aten.clamp_min,
        aten.clamp_max,
        aten.linear_backward,
        aten._fused_rms_norm,
        aten._fused_rms_norm_backward,
        aten.fill.Scalar,
        aten.fill.Tensor,
        aten.index_add,
    ]
)


def register_decomposition(
    ops: torch._ops.OperatorBase | list[torch._ops.OperatorBase],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    for op in ops if isinstance(ops, list) else [ops]:
        # Guards against duplicates
        if op in neuron_decompositions:
            raise ValueError(f"Duplicate decomposition registration for: {op}")
    return decomp.register_decomposition(ops, neuron_decompositions)


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


@register_decomposition([aten.as_strided.default])
def as_strided(input_tensor, size, stride, storage_offset=0):
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


@register_decomposition([aten.topk.default])
def topk_decomposition(x, k, dim=-1, largest=True, sorted=True):
    # Normalize dim to positive index
    if dim < 0:
        dim = x.ndim + dim

    # If dim is already the last dimension and parameters are good, don't decompose
    if dim == x.ndim - 1 and largest and sorted:
        return NotImplemented

    # Handle non-last dimension by rearranging
    if dim != x.ndim - 1:
        # Move target dim to last position
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_transposed = x.permute(perm)

        # Apply topk on last dimension
        if largest:
            values, indices = torch.topk(x_transposed, k, -1, True)
        else:
            values, indices = torch.topk(-x_transposed, k, -1, True)
            values = -values

        # Move dimension back to original position
        values = values.permute(perm)
        indices = indices.permute(perm)
        return values, indices

    # Handle last dimension cases
    elif not largest:
        values, indices = torch.topk(-x, k, dim=-1, largest=True, sorted=True)
        return -values, indices
    else:
        # largest=True but sorted=False and dim=-1, force sorted=True
        return torch.topk(x, k, dim=-1, largest=True, sorted=True)


def scaled_dot_product_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    philox_seed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposition of scaled_dot_product_attention into basic ops.

    Args:
        query: (B, H, L, E) or (B, L, H, E) depending on layout
        key: (B, H, S, E) or (B, S, H, E)
        value: (B, H, S, E) or (B, S, H, E)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking
        scale: Optional scale factor (default: 1/sqrt(E))
        enable_gqa: Enable grouped-query attention

    Returns:
        Attention output tensor, lse
    """
    # Get dimensions - assume (B, H, L, E) layout
    B, q_heads, L, E = query.shape
    _, kv_heads, S, _ = key.shape

    # Handle GQA
    if enable_gqa and q_heads > kv_heads and q_heads % kv_heads == 0:
        repeat_factor = q_heads // kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

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

    max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_scores)
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
    attn_weights = exp_scores / sum_exp

    # Compute LSE
    lse = max_scores.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    lse = lse.to(torch.float32)

    # TODO: Add dropout support
    # Need to generate consistent mask across fwd and bwd
    # philox_seed will have to be materialized in the bwd to generate the same mask
    # which makes the graph data-dependent
    if dropout_p > 0.0:
        attn_weights, _ = torch.ops.aten.native_dropout.default(attn_weights, dropout_p, train=True)

    output = torch.matmul(attn_weights, value)

    return (output, lse)


@register_decomposition([aten._scaled_dot_product_fused_attention_overrideable.default])
def sdpa_fused_overrideable(
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
    # Force training=True, since torch.is_grad_enabled() remains False
    training = True
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
    can_use_nki_sdpa = can_use_nki_sdpa and nki_attention_enabled

    # Detect GQA
    _, kv_heads, _, _ = key.shape
    is_gqa = kv_heads < H and H % kv_heads == 0
    if kv_heads < H and H % kv_heads != 0:
        raise RuntimeError(
            "Number of heads in key and value must divide the number of heads in query"
        )

    # Create config with custom seq_tile_size
    config = FlashConfig(training=training, seq_tile_size=seq_tile_size, should_transpose_v=False)

    if dropout_p > 0.0:
        philox_seed = torch.randint(0, 2**31 - 1, (1,), device=query.device, dtype=torch.int32)
        philox_offset = torch.zeros(1, device=query.device, dtype=torch.int32)
    else:
        philox_seed = (query.flatten()[0] * 0).to(torch.int32)
        philox_offset = (query.flatten()[0] * 0).to(torch.int32)

    if can_use_nki_sdpa and nki_attention_enabled:
        try:
            # Handle GQA
            if is_gqa:
                repeat_factor = H // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            seed = (
                philox_seed
                if dropout_p > 0.0 and philox_seed is not None and philox_seed.numel() > 0
                else None
            )
            q = query.transpose(2, 3)
            k = key.transpose(2, 3)
            v = value
            grid_heads = H

            output, logsumexp = wrapped_flash_fwd[B, grid_heads](
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
        except Exception as e:
            logger.debug("SDPA forward NKI kernel call failed with error:", e)
            pass

    # Fallback to manual decomposition
    output, logsumexp = scaled_dot_product_attention_decomposition(
        query,
        key,
        value,
        attn_bias,
        dropout_p,
        is_causal,
        scale,
        enable_gqa=True,
        philox_seed=philox_seed,
    )

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


def _can_use_nki_flash_attention_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lse: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> bool:
    """Check NKI flash attention contraints"""
    if not _getenv_bool("TORCH_NEURONX_ENABLE_NKI_SDPA", True):
        return False

    if attn_bias is not None:
        return False

    B, q_heads, L, E = query.shape
    _, kv_heads, S, _ = key.shape
    if key.shape != value.shape:
        return False
    if L % 512 != 0 or S % 512 != 0:
        return False
    return not E > 128


def compute_attention_backward_decomposition(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    is_causal: bool,
    dropout_p: float,
    mixed_precision: bool,
    philox_seed: torch.Tensor | None = None,
):
    """Torch decomposition for attention backward compute"""
    orig_dtype = query.dtype
    acc_dtype = torch.float32 if mixed_precision else orig_dtype

    grad_out = grad_out.to(acc_dtype)
    query = query.to(acc_dtype)
    key = key.to(acc_dtype)
    value = value.to(acc_dtype)
    lse = lse.to(acc_dtype)
    if attn_bias.numel() > 1:
        attn_bias = attn_bias.to(acc_dtype)

    logits = query @ key.transpose(-2, -1) * scale

    if is_causal:
        seq_len_q, seq_len_k = logits.shape[-2:]
        temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        logits = logits.masked_fill(temp_mask.logical_not(), float("-inf"))
    elif attn_bias.numel() > 1:
        logits = logits + attn_bias

    probs = torch.exp(logits - lse.unsqueeze(-1))

    # TODO: Add dropout support
    # Need to generate consistent mask across fwd and bwd
    # philox_seed will have to be materialized in the bwd to generate the same mask
    # which makes the graph data-dependent
    if dropout_p > 0.0:
        pass

    grad_v = probs.transpose(-2, -1) @ grad_out
    grad_out_v = grad_out @ value.transpose(-2, -1)
    sum_reduction = (grad_out_v * probs).sum(dim=-1, keepdim=True)
    ds = probs * (grad_out_v - sum_reduction) * scale

    if is_causal:
        seq_len_q, seq_len_k = ds.shape[-2:]
        temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        ds = ds.masked_fill(temp_mask.logical_not(), 0.0)

    grad_q = ds @ key
    grad_k = ds.transpose(-2, -1) @ query

    grad_attn_bias = ds.to(orig_dtype) if attn_bias.numel() > 1 else attn_bias.to(orig_dtype)

    grad_q = grad_q.to(orig_dtype)
    grad_k = grad_k.to(orig_dtype)
    grad_v = grad_v.to(orig_dtype)

    return grad_q, grad_k, grad_v, grad_attn_bias


def _prepare_backward_inputs(query, key, value, orig_k_shape, scale):
    """Prepare common inputs for backward pass."""
    B, q_heads, L, E = query.shape
    _, kv_heads, S, _ = key.shape
    orig_kv_heads = orig_k_shape[1]

    # Calculate scale if not provided
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    # Handle GQA
    is_gqa = q_heads > orig_kv_heads and q_heads % orig_kv_heads == 0
    if is_gqa and kv_heads == orig_kv_heads:
        repeat_factor = q_heads // orig_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    return B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value


def _reduce_gqa_gradients(grad_k, grad_v, is_gqa, b, orig_kv_heads, q_heads, s, e):
    """Reduce gradients for GQA back to original KV shapes."""
    if is_gqa:
        repeat_factor = q_heads // orig_kv_heads
        grad_k = grad_k.view(b, orig_kv_heads, repeat_factor, s, e).sum(dim=2)
        grad_v = grad_v.view(b, orig_kv_heads, repeat_factor, s, e).sum(dim=2)
    return grad_k, grad_v


@register_decomposition(
    [torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward.default]
)
def scaled_dot_product_fused_attention_overrideable_backward_decomposition(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    grad_input_mask: list[bool] | None = None,
    out: torch.Tensor | None = None,
    logsumexp: torch.Tensor | None = None,
    cum_seq_q: torch.Tensor | None = None,
    cum_seq_k: torch.Tensor | None = None,
    max_q: int | None = None,
    max_k: int | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    philox_seed: torch.Tensor | None = None,
    philox_offset: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Decomposition for scaled_dot_product_fused_attention_overrideable backward."""
    can_use_nki = _can_use_nki_flash_attention_backward(
        grad_out, query, key, value, logsumexp, attn_bias, dropout_p
    )
    orig_k_shape = key.shape

    # Prepare common inputs
    B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value = _prepare_backward_inputs(
        query, key, value, orig_k_shape, scale
    )
    # Try NKI
    if can_use_nki:
        try:
            # Transform inputs to NKI format
            q_nki = query.transpose(-2, -1)
            k_nki = key.transpose(-2, -1)
            v_nki = value.transpose(-2, -1)
            out_nki = out.transpose(-2, -1) if out is not None else None
            grad_out_nki = grad_out.transpose(-2, -1)

            # reshape LSE for NKI
            if logsumexp.ndim == 3:
                tile_size = 128
                n_tiles = L // tile_size
                if L % tile_size == 0:
                    lse_4d = logsumexp.view(B, q_heads, n_tiles, tile_size).transpose(-2, -1)
                else:
                    raise ValueError(
                        f"Sequence length {L} must be divisible by tile size {tile_size}"
                    )
            else:
                lse_4d = logsumexp

            # Prepare seed and grid
            seed = (
                philox_seed
                if dropout_p > 0.0 and philox_seed is not None and philox_seed.numel() > 0
                else None
            )
            logical_neuron_cores = int(get_logical_neuron_cores())
            if logical_neuron_cores > 1 and q_heads % logical_neuron_cores == 0:
                import neuronxcc.nki.language as nl

                grid = (B, nl.nc(logical_neuron_cores) * (q_heads // logical_neuron_cores))
            else:
                grid = (B, q_heads)

            seed_1d = seed.view(-1)[0:1] if seed is not None and seed.numel() > 0 else None

            # Call NKI bwd kernel
            grad_q_nki, grad_k_nki, grad_v_nki = wrapped_flash_bwd[grid](
                q_nki,
                k_nki,
                v_nki,
                out_nki,
                grad_out_nki,
                lse_4d,
                seed_1d,
                use_causal_mask=is_causal,
                mixed_precision=True,
                dropout_p=dropout_p,
                softmax_scale=scale,
            )

            # Transform gradients back to ATen format
            grad_q = grad_q_nki.transpose(-2, -1)
            grad_k = grad_k_nki.transpose(-2, -1)
            grad_v = grad_v_nki.transpose(-2, -1)

            # Reduce gradients for GQA
            grad_k, grad_v = _reduce_gqa_gradients(
                grad_k, grad_v, is_gqa, B, orig_kv_heads, q_heads, S, E
            )

            return grad_q, grad_k, grad_v, None

        except Exception as e:
            logger.debug("SDPA backward NKI kernel call failed with exception:", e)
            pass

    # Fallback to manual backward decomposition
    if attn_bias is None:
        attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)

    grad_q, grad_k, grad_v, grad_attn_bias = compute_attention_backward_decomposition(
        grad_out,
        query,
        key,
        value,
        attn_bias,
        logsumexp,
        scale,
        is_causal,
        dropout_p,
        True,
        philox_seed,
    )

    # Reduce gradients for GQA
    grad_k, grad_v = _reduce_gqa_gradients(grad_k, grad_v, is_gqa, B, orig_kv_heads, q_heads, S, E)

    # Handle grad_attn_bias
    if grad_attn_bias.numel() == 1:
        grad_attn_bias = None

    return grad_q, grad_k, grad_v, grad_attn_bias


@register_decomposition([aten.scalar_tensor.default])
def scalar_tensor(
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


@register_decomposition([aten.index_copy.default])
def index_copy(
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


def nonzero_with_count(
    tensor: torch.Tensor,
    size: int,
    fill_value: int = -1,
    with_count: bool = False,
) -> tuple[torch.Tensor, int]:
    """Custom decomposition for nonzero and nonzero_static

    Args:
        tensor: input tensor
        size: fixed size on output tensor. If there are fewer nonzero elements
            than the specified size, then indices will be padded with ``fill_value``
        fill_value: padding value
        with_count: return number of nonzero element or not. Default to False to support
            nonzero_static. When set to True, will allow torch.nonzero to do a slice
            on the output hence avoid recompilation.

    Reference https://github.com/jax-ml/jax/blob/43d1d0a177f0cc79977e84b2e648fab560531e08/jax/_src/numpy/lax_numpy.py#L3624
    """
    # Match safe_nonzero JAX implementation exactly
    mask = tensor != 0
    num_nonzero = mask.sum()

    if size is None:
        size = num_nonzero

    if tensor.numel() == 0 or size == 0:
        coords = tuple(
            torch.zeros(size, dtype=torch.long, device=tensor.device) for _ in tensor.shape
        )
        result = torch.stack(coords, dim=1)
        if with_count:
            return result, num_nonzero
        else:
            return result

    # Cumsum counts - this gives position in nonzero sequence
    cumsum_counts = torch.cumsum(mask.flatten().float(), dim=0)

    # Ignore OOB values and create weights
    in_range = cumsum_counts < size
    cumsum_counts = torch.clamp(cumsum_counts, 0, size - 1).to(torch.long)
    weights = torch.where(in_range, 1.0, 0.0)

    # Use scatter_add (functional) instead of scatter_add_ (in-place)
    bin_counts = torch.zeros(size, dtype=weights.dtype, device=tensor.device)
    bin_counts = torch.scatter_add(bin_counts, 0, cumsum_counts, weights)

    # Cumsum to get flat indices
    flat_indices = torch.cumsum(bin_counts, dim=0).to(torch.long)

    # Create strides dynamically without hardcoded constants
    shape = tensor.shape
    strides = []
    for i in range(len(shape)):
        if i + 1 < len(shape):
            stride = 1
            for dim in shape[i + 1 :]:
                stride *= dim
        else:
            stride = 1
        strides.append(stride)

    # Convert flat indices to coordinates
    coords = []
    for stride in strides:
        coord = (flat_indices // stride) % shape[len(coords)]
        coords.append(coord)

    # Apply fill_value if provided
    if fill_value is not None:
        fill_mask = torch.arange(size, device=tensor.device) >= num_nonzero
        if isinstance(fill_value, (list, tuple)):  # noqa: UP038
            # Multiple fill values for each dimension
            for i, (coord, fval) in enumerate(zip(coords, fill_value, strict=False)):
                coords[i] = torch.where(fill_mask, fval, coord)
        else:
            # Single fill value for all dimensions
            for i, coord in enumerate(coords):
                coords[i] = torch.where(fill_mask, fill_value, coord)

    result = torch.stack(coords, dim=1)
    if with_count:
        return result, num_nonzero
    else:
        return result


@register_decomposition([aten.nonzero_static.default])
def nonzero_static_decomposition(
    tensor: torch.Tensor,
    size: int,
    fill_value: int = -1,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    return nonzero_with_count(tensor, size=size, fill_value=fill_value)


@register_decomposition([aten.linalg_vector_norm.default])
def linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    """Custom decomposition for vector_norm to ensure correct results.

    Reference https://github.com/pytorch/pytorch/blob/bfa6f5e0730dead84017e779e02de6cea768ee33/torch/_refs/linalg/__init__.py#L131
    """
    # Check input dtype
    if not (x.dtype.is_floating_point or x.dtype.is_complex):
        raise RuntimeError(
            "linalg.vector_norm: Expected a floating point or complex tensor as input. "
            f"Got {x.dtype}"
        )

    # Check dtype parameter
    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise RuntimeError(
                f"linalg.vector_norm: dtype should be floating point or complex. Got {dtype}"
            )

        if dtype.is_complex != x.dtype.is_complex:
            expected = "complex" if x.dtype.is_complex else "real"
            raise RuntimeError(
                f"linalg.vector_norm: dtype should be {expected} for {expected} inputs. Got {dtype}"
            )

        # Check for narrowing conversion
        dtype_size = torch.finfo(dtype).bits
        x_dtype_size = torch.finfo(x.dtype).bits
        if dtype_size < x_dtype_size:
            raise RuntimeError(
                f"linalg.vector_norm: the dtype of the input ({x.dtype}) should be convertible "
                f"without narrowing to the specified dtype ({dtype})"
            )

    if dtype is not None:
        x = x.to(dtype)

    if ord == 0.0:
        return torch.sum(torch.ne(x, 0.0), dim=dim, keepdim=keepdim, dtype=dtype)
    elif ord == float("inf"):
        result = torch.amax(torch.abs(x), dim=dim, keepdim=keepdim)
    elif ord == float("-inf"):
        result = torch.amin(torch.abs(x), dim=dim, keepdim=keepdim)
    else:
        x_abs = torch.abs(x)
        result = torch.pow(torch.sum(torch.pow(x_abs, ord), dim=dim, keepdim=keepdim), 1.0 / ord)

        # Handle negative ord with zeros
        if ord < 0:
            has_zero = torch.sum(torch.eq(x, 0), dim=dim, keepdim=keepdim) > 0
            result = torch.where(has_zero, torch.zeros_like(result), result)

    # Handle NaNs
    has_nan = torch.sum(torch.isnan(x), dim=dim, keepdim=keepdim) > 0
    result = torch.where(has_nan, torch.full_like(result, float("nan")), result)

    return result


@register_decomposition([aten.sigmoid_backward.default])
def sigmoid_backward_decomposition(grad_output, output):
    """Decomposition for sigmoid_backward into primitive ops"""
    return grad_output * (output * (1 - output))


@register_decomposition([aten.linear_backward])
def linear_backward_decomposition(input_, grad_output_, weight_, output_mask):
    """
    Decompose linear backwards pass (linear forward pass: y = xA^t + b)

    :param input_: input (x)
    :param grad_output_: output gradient (dy)
    :param weight_: weight (A)
    :param output_mask: mask of 3 bools:
        output_mask[0] - whether to return input gradient
        output_mask[1] - whether to return weight gradient
        output_mask[2] - whether to return bias gradient
    """

    grad_input = None
    grad_weight = None
    grad_bias = None
    if output_mask[0]:
        grad_input = torch.matmul(grad_output_, weight_)
    if output_mask[1]:
        contract_dims = [tuple(range(grad_output_.ndim - 1)), tuple(range(grad_output_.ndim - 1))]
        grad_weight = torch.tensordot(grad_output_, input_, dims=contract_dims)
    if output_mask[2]:
        if grad_output_.ndim == 1:
            grad_bias = grad_output_.clone()
        else:
            dims_to_reduce = tuple(range(grad_output_.ndim - 1))
            grad_bias = torch.sum(grad_output_, dim=dims_to_reduce, keepdim=False)
    return (grad_input, grad_weight, grad_bias)


def get_decomposition_table(decompose_all: bool = False) -> dict:
    """
    Get the complete decomposition table for the Neuron backend.

    Returns:
        Decomposition table for the backend
    """
    return neuron_decompositions
