# ruff: noqa: N806, E402, F841
"""
Custom decompositions for torch.compile neuron backend

This module contains custom decompositions that avoid operations
that cause issues in the torch-mlir to StableHLO conversion pipeline.

See docs/decompositions.md for a guide on adding new decompositions.
See https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py
for PyTorch's built-in decompositions.
"""

import logging
import math
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import torch
import torch._decomp as decomp
import torch.nn.functional as F  # noqa: N812
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
        aten.logsumexp,
        aten.squeeze.dims,
        aten.binary_cross_entropy,
        aten.native_batch_norm_backward,
        aten.nansum,
        aten.addr,
        aten._unsafe_index,
        aten._unsafe_index_put,
        aten.split,
        aten.split_with_sizes,
        aten.unbind,
        aten.replication_pad1d,
        aten.replication_pad2d,
        aten.replication_pad3d,
        aten.reflection_pad1d,
        aten.reflection_pad2d,
        aten.reflection_pad3d,
        aten.embedding,
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
        aten.native_dropout_backward,
        aten.gelu_backward,
        aten.threshold_backward,
        aten._log_softmax_backward_data,
        aten._softmax_backward_data,
        aten.select_backward,
        aten.slice_backward,
        aten.softplus_backward,
        aten.index_select_backward,
        aten.col2im,  # accuracy mismatch on neuron, pending NCF-486
        aten.lerp,
        aten.elu,
        aten.elu_backward,
        aten.silu,
    ]
)


def register_decomposition(
    ops: torch._ops.OperatorBase | list[torch._ops.OperatorBase],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Register a decomposition function for one or more ATen operators.

    Decorator that registers a function as the decomposition for the specified
    operators in the Neuron decomposition table. Guards against duplicate
    registrations.

    Args:
        ops (torch._ops.OperatorBase | list[torch._ops.OperatorBase]): Single
            operator or list of operators to decompose.

    Returns:
        Callable: Decorator that registers the decorated function.

    Raises:
        ValueError: If any operator is already registered in neuron_decompositions.
    """
    for op in ops if isinstance(ops, list) else [ops]:
        # Guards against duplicates
        if op in neuron_decompositions:
            raise ValueError(f"Duplicate decomposition registration for: {op}")
    return decomp.register_decomposition(ops, neuron_decompositions)


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
    """Decompose topk for non-standard dimension or parameter combinations.

    Handles cases where dim is not the last dimension or when largest=False
    by transposing, applying topk, and transposing back.

    Args:
        x (torch.Tensor): Input tensor.
        k (int): Number of top elements to return.
        dim (int): Dimension to sort along. Defaults to -1.
        largest (bool): If True, return largest elements. Defaults to True.
        sorted (bool): If True, return sorted elements. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (values, indices) tensors.
    """
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
    """Decompose scaled_dot_product_attention into basic ops.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_mask (torch.Tensor | None): Optional attention mask.
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether to apply causal masking.
        scale (float | None): Scale factor. Defaults to 1/sqrt(E).
        enable_gqa (bool): Enable grouped-query attention.
        philox_seed (torch.Tensor | None): Random seed for dropout.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (attention_output, logsumexp).
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

    dropout_mask = None
    if dropout_p > 0:
        attn_weights, dropout_mask = torch.ops.aten.native_dropout.default(
            attn_weights, dropout_p, train=True
        )

    output = torch.matmul(attn_weights, value)

    return (output, lse, dropout_mask)


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
    """Decompose _scaled_dot_product_fused_attention_overrideable.

    Routes to NKI SDPA kernel when possible, otherwise falls back to
    the standard decomposition.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, D).
        value (torch.Tensor): Value tensor of shape (B, H, S, D).
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether to apply causal masking.
        return_debug_mask (bool): Whether to return debug mask.
        scale (float | None): Scale factor. Defaults to 1/sqrt(E).

    Returns:
        tuple: (attention_output, logsumexp, philox_seed, philox_offset, debug_mask).
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
    can_use_nki_sdpa = can_use_nki_sdpa and dropout_p == 0

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
    output, logsumexp, dropout_mask = scaled_dot_product_attention_decomposition(
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

    # directly pass the dropout mask to the backward FX graph
    # via the philox_seed parameter
    philox_seed = dropout_mask

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
    """Check if NKI flash attention backward kernel can be used.

    Args:
        grad_out (torch.Tensor): Gradient of attention output.
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        lse (torch.Tensor): Logsumexp from forward pass.
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.

    Returns:
        bool: True if NKI kernel constraints are satisfied.
    """
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
    if dropout_p > 0:
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
    """Compute attention backward pass using torch decomposition.

    Args:
        grad_out (torch.Tensor): Gradient of attention output.
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor): Attention bias tensor.
        lse (torch.Tensor): Logsumexp from forward pass.
        scale (float): Attention scale factor.
        is_causal (bool): Whether to apply causal masking.
        dropout_p (float): Dropout probability.
        mixed_precision (bool): Whether to use mixed precision.
        philox_seed (torch.Tensor | None): Random seed for dropout.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (grad_query, grad_key, grad_value).
    """
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

    # use the dropout mask passed to the backwards FX graph
    # via the philox_seed parameter
    dropout_mask = philox_seed
    masked_probs = probs
    if dropout_mask is not None:
        if dropout_p == 1:
            masked_probs = probs * dropout_mask
        else:
            masked_probs = probs * dropout_mask * (1 / (1 - dropout_p))

    grad_v = masked_probs.transpose(-2, -1) @ grad_out
    grad_out_v = grad_out @ value.transpose(-2, -1)
    grad_out_mask = grad_out_v
    if dropout_mask is not None:
        if dropout_p == 1:
            grad_out_mask = grad_out_v * dropout_mask
        else:
            grad_out_mask = grad_out_v * dropout_mask * (1 / (1 - dropout_p))

    sum_reduction = (grad_out_mask * probs).sum(dim=-1, keepdim=True)
    ds = probs * (grad_out_mask - sum_reduction) * scale

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
    """Prepare common inputs for attention backward pass.

    Handles GQA key/value expansion and scale calculation.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H_kv, S, E).
        value (torch.Tensor): Value tensor of shape (B, H_kv, S, E).
        orig_k_shape (tuple): Original key shape before any expansion.
        scale (float | None): Scale factor, computed if None.

    Returns:
        tuple: (B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value).
    """
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
    """Reduce GQA gradients back to original KV shapes.

    Args:
        grad_k (torch.Tensor): Key gradient tensor.
        grad_v (torch.Tensor): Value gradient tensor.
        is_gqa (bool): Whether grouped-query attention was used.
        b (int): Batch size.
        orig_kv_heads (int): Original number of KV heads.
        q_heads (int): Number of query heads.
        s (int): Sequence length.
        e (int): Head dimension.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (grad_k, grad_v) with original shapes.
    """
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
    """Decompose scaled_dot_product_fused_attention_overrideable backward.

    Routes to NKI kernel when possible, otherwise uses torch decomposition.

    Args:
        grad_out (torch.Tensor): Gradient of attention output.
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor | None): Optional attention bias.
        grad_input_mask (list[bool] | None): Mask for which gradients to compute.
        out (torch.Tensor | None): Forward pass output.
        logsumexp (torch.Tensor | None): Logsumexp from forward pass.
        cum_seq_q (torch.Tensor | None): Cumulative sequence lengths for queries.
        cum_seq_k (torch.Tensor | None): Cumulative sequence lengths for keys.
        max_q (int | None): Maximum query sequence length.
        max_k (int | None): Maximum key sequence length.
        dropout_p (float): Dropout probability.
        is_causal (bool): Whether causal masking was applied.
        philox_seed (torch.Tensor | None): Random seed from forward pass.
        philox_offset (torch.Tensor | None): Random offset from forward pass.
        scale (float | None): Scale factor used in forward pass.

    Returns:
        tuple: (grad_query, grad_key, grad_value, grad_attn_bias).
    """
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
    """Decompose scalar_tensor to full() for better StableHLO support.

    Args:
        value: Scalar value to convert to tensor.
        dtype: Optional dtype for the tensor.
        layout: Tensor layout (unused).
        device: Target device.
        pin_memory: Whether to pin memory (unused).

    Returns:
        torch.Tensor: 0-dimensional tensor containing the scalar value.
    """
    # Use full with empty shape to create a scalar tensor
    return torch.full((), value, dtype=dtype, device=device)


@register_decomposition([aten.index_copy.default])
def index_copy(
    input: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """Decompose index_copy into functional scatter.

    Args:
        input (torch.Tensor): Input tensor to copy into.
        dim (int): Dimension along which to index.
        index (torch.Tensor): Indices of elements to copy.
        source (torch.Tensor): Source tensor to copy from.

    Returns:
        torch.Tensor: Result with source values scattered into input.
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
    """Find nonzero elements with fixed output size.

    Args:
        tensor (torch.Tensor): Input tensor to find nonzero elements.
        size (int): Fixed output size. Pads with fill_value if fewer nonzeros.
        fill_value (int): Padding value for unused slots. Defaults to -1.
        with_count (bool): If True, return count of nonzero elements.

    Returns:
        tuple[torch.Tensor, int]: (indices, count) if with_count, else just indices.
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
    """Decompose nonzero_static to nonzero_with_count.

    Args:
        tensor (torch.Tensor): Input tensor to find nonzero elements.
        size (int): Maximum number of nonzero elements to return.
        fill_value (int): Value to fill unused slots. Defaults to -1.
        out (torch.Tensor | None): Optional output tensor.

    Returns:
        tuple[torch.Tensor, int]: (indices, count) of nonzero elements.
    """
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


def index_put_decomposition(input, indices, values, accumulate=False):
    """
    Decomposition of aten.index_put that handles negative indices and boolean masks

    This function decomposes index_put into simpler operations (torch.where or torch.scatter)

    Args:
        input (torch.Tensor): The tensor to be modified.
        indices (tuple[torch.Tensor | None, ...]): A tuple of index tensors or None values.
            - None represents a full slice (:) for that dimension
            - Boolean tensors are converted to integer indices via nonzero()
            - Integer tensors specify explicit indices (negative values supported)
            - Lists/tuples are converted to tensors
        values (torch.Tensor): Values to place into the tensor at the specified indices.
            Must be broadcastable to the shape of the indexed region.
        accumulate (bool, optional): If True, values are added to existing values
            using scatter_add. If False, values replace existing values using scatter.
            Defaults to False.

    Returns:
        torch.Tensor: A new tensor with values placed at the specified indices.

    Raises:
        IndexError: If indices are out of bounds after negative index normalization.
        IndexError: If index tensors cannot be broadcast together.
        IndexError: If values cannot be broadcast to the indexed region shape.
    """

    def try_get_where_mask(x, indices, values):
        """
        Returns prepared mask if operation can use torch.where
        and returns None otherwise
        """
        if values.numel() != 1 or accumulate:
            return None

        num_defined_indices = 0
        mask = None

        for index in indices:
            if index is not None:
                if hasattr(index, "dtype") and index.dtype == torch.bool:
                    if mask is not None:
                        return None
                    mask = index
                    for j in range(index.ndim):
                        if index.shape[j] != x.shape[num_defined_indices + j]:
                            return None
                    num_defined_indices += index.ndim
                else:
                    return None
            else:
                num_defined_indices += 1

        if mask is None:
            return None

        if mask.ndim < x.ndim:
            for _ in range(x.ndim - mask.ndim):
                mask = mask.unsqueeze(-1)

        return mask

    if (mask := try_get_where_mask(input, indices, values)) is not None:
        return torch.where(mask, values.to(input.dtype), input)

    if (
        any(isinstance(idx, torch.Tensor) and idx.dtype == torch.bool for idx in indices)
        and values.numel() == 1
    ):
        bool_info = None
        int_info = None

        for i, idx in enumerate(indices):
            if isinstance(idx, torch.Tensor):
                if idx.dtype == torch.bool and idx.dim() == 1:
                    bool_info = (i, idx)
                elif idx.dtype in (torch.int32, torch.int64):
                    int_info = (i, idx)

        if bool_info and int_info:
            bool_dim, bool_idx = bool_info
            int_dim, int_idx = int_info
            int_idx = torch.where(int_idx < 0, int_idx + input.shape[int_dim], int_idx)

            k = bool_idx.sum()
            m = int_idx.numel()

            arange = torch.arange(input.shape[int_dim], device=input.device)

            # broadcast mask: for k==1 or m==1 cases
            int_mask = torch.zeros(input.shape[int_dim], dtype=torch.bool, device=input.device)
            int_mask = int_mask.scatter(0, int_idx.flatten(), True)
            mask_broadcast = bool_idx.unsqueeze(1) & int_mask.unsqueeze(0)

            # paired mask: for k==m case
            cumsum = torch.cumsum(bool_idx.to(torch.int64), dim=0) - 1
            cumsum = torch.clamp(cumsum, min=0, max=m - 1)
            int_idx_selected = int_idx.flatten()[cumsum]  # index, not expand
            mask_paired = bool_idx.unsqueeze(1) & (arange == int_idx_selected.unsqueeze(1))

            # select mask based on k == m
            mask = torch.where(k == m, mask_paired, mask_broadcast)

            if bool_dim > int_dim:
                mask = mask.T

            for _ in range(2, input.dim()):
                mask = mask.unsqueeze(-1)
            mask = mask.expand(input.shape)

            values = values.to(input.dtype).expand(input.shape)
            if accumulate:
                return torch.where(mask, input + values, input)
            return torch.where(mask, values, input)

    def convert_indices(indices, device):
        for index in indices:
            if index is None:
                yield index
            elif isinstance(index, list | tuple):
                yield torch.tensor(index, device=device)
            elif isinstance(index, torch.Tensor) and index.dtype == torch.bool:
                raise RuntimeError(
                    "Boolean index_put is not supported in torch.compile on Neuron. "
                    "Boolean indices require nonzero() which produces dynamic shapes."
                )
            else:
                yield index

    converted_indices = convert_indices(indices, input.device)

    # normalize negative indices
    normalized_indices = []
    for i, idx in enumerate(converted_indices):
        if idx is None:
            normalized_indices.append(None)
        elif isinstance(idx, torch.Tensor):
            if idx.dtype in (torch.int32, torch.int64):
                dim_size = input.shape[i]
                idx = torch.where(idx < 0, idx + dim_size, idx)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            normalized_indices.append(idx)
        else:
            normalized_indices.append(idx)

    # separate indexed dims from kept dims
    indexed_dims = []  # list of (dim_index, index_tensor)
    for i, idx in enumerate(normalized_indices):
        if idx is not None:
            indexed_dims.append((i, idx))

    if len(indexed_dims) == 0:
        return input

    indexed_positions = [d for d, _ in indexed_dims]
    indexed_dim_set = set(indexed_positions)
    kept_dims = [d for d in range(input.dim()) if d not in indexed_dim_set]

    # broadcast index tensors
    idx_tensors = [idx for _, idx in indexed_dims]

    if len(idx_tensors) > 1:
        try:
            broadcast_shape = torch.broadcast_shapes(*[t.shape for t in idx_tensors])
        except RuntimeError as e:
            shapes_str = ", ".join([str(list(t.shape)) for t in idx_tensors])
            raise IndexError(
                f"shape mismatch: indexing tensors could not be broadcast together "
                f"with shapes {shapes_str}"
            ) from e
        broadcasted = [t.expand(broadcast_shape) for t in idx_tensors]
    else:
        broadcast_shape = list(idx_tensors[0].shape)
        broadcasted = idx_tensors

    broadcast_shape = list(broadcast_shape)

    # determine output shape based on indexing pattern
    # rules:
    # i. Adjacent indexed dims: broadcast shape replaces them in-place
    # ii. Non-adjacent indexed dims: broadcast shape goes to front

    are_adjacent = (
        all(
            indexed_positions[i + 1] - indexed_positions[i] == 1
            for i in range(len(indexed_positions) - 1)
        )
        if len(indexed_positions) > 1
        else True
    )

    if are_adjacent:
        # build shape with broadcast dims at position of first indexed dim
        expected_shape = []
        broadcast_inserted = False
        for d in range(input.dim()):
            if d in indexed_dim_set:
                if not broadcast_inserted:
                    expected_shape.extend(broadcast_shape)
                    broadcast_inserted = True
                # skip subsequent indexed dims
            else:
                expected_shape.append(input.shape[d])
        first_indexed_pos = indexed_positions[0]
    else:
        # non-adjacent: broadcast shape at front, then kept dims
        kept_shape = [input.shape[d] for d in kept_dims]
        expected_shape = broadcast_shape + kept_shape
        # broadcast dims are at front
        first_indexed_pos = 0

    # broadcast values to expected shape
    values = values.to(input.dtype)

    if values.numel() == 1:
        values = values.expand(expected_shape)
    elif list(values.shape) == expected_shape:
        pass
    else:
        try:
            values = values.expand(expected_shape)
        except RuntimeError:
            raise IndexError(
                f"shape mismatch: value tensor of shape {list(values.shape)} "
                f"cannot be broadcast to indexing result of shape {expected_shape}"
            ) from None

    # compute strides
    strides = []
    for i in range(input.dim()):
        stride = 1
        for j in range(i + 1, input.dim()):
            stride *= input.shape[j]
        strides.append(stride)

    # build linear index tensor
    flat_idx = torch.zeros(expected_shape, dtype=torch.int64, device=input.device)

    # add contribution from indexed dimensions
    for idx_num, (dim, _) in enumerate(indexed_dims):
        idx_tensor = broadcasted[idx_num]

        # reshape index tensor to align with expected_shape
        if are_adjacent:
            # broadcast dims are at first_indexed_pos
            # idx_tensor has shape broadcast_shape, needs to be at position first_indexed_pos
            shape = [1] * len(expected_shape)
            for i, s in enumerate(broadcast_shape):
                shape[first_indexed_pos + i] = s
        else:
            # broadcast dims are at front
            shape = [1] * len(expected_shape)
            for i, s in enumerate(broadcast_shape):
                shape[i] = s

        idx_reshaped = idx_tensor.view(shape).expand(expected_shape)
        flat_idx = flat_idx + idx_reshaped.to(torch.int64) * strides[dim]

    # add contribution from kept dimensions (full slices)
    for kept_idx, dim in enumerate(kept_dims):
        arange = torch.arange(input.shape[dim], dtype=torch.int64, device=input.device)

        # find position of this kept dim in expected_shape
        if are_adjacent:
            # count kept dims before this one
            kept_before = sum(1 for d in kept_dims[:kept_idx] if d < first_indexed_pos)
            # position is either before broadcast (if dim < first_indexed_pos)
            # or after broadcast (if dim > last indexed pos)
            if dim < first_indexed_pos:
                pos = kept_before
            else:
                pos = first_indexed_pos + len(broadcast_shape) + (kept_idx - kept_before)
        else:
            # kept dims come after broadcast dims
            pos = len(broadcast_shape) + kept_idx

        shape = [1] * len(expected_shape)
        shape[pos] = input.shape[dim]
        arange = arange.view(shape)
        flat_idx = flat_idx + arange * strides[dim]

    # handle empty indices
    if flat_idx.numel() == 0:
        return input

    # flatten and scatter
    flat_input = input.flatten()
    flat_values = values.flatten()
    flat_idx = flat_idx.flatten()

    if accumulate:
        result = torch.scatter_add(flat_input, 0, flat_idx, flat_values)
    else:
        result = torch.scatter(flat_input, 0, flat_idx, flat_values)

    return result.reshape(input.shape)


@register_decomposition([aten.sigmoid_backward.default])
def sigmoid_backward_decomposition(grad_output, output):
    """Decompose sigmoid_backward into primitive ops.

    Args:
        grad_output (torch.Tensor): Gradient of the output.
        output (torch.Tensor): Output from forward pass (sigmoid(x)).

    Returns:
        torch.Tensor: Gradient with respect to input.
    """
    return grad_output * (output * (1 - output))


@register_decomposition([aten.silu_backward.default])
def silu_backward_decomposition(grad_output, x):
    """Decompose silu_backward into primitive ops.

    Args:
        grad_output (torch.Tensor): Gradient of the output.
        x (torch.Tensor): Input tensor from forward pass.

    Returns:
        torch.Tensor: Gradient with respect to input.
    """
    sigmoid_x = torch.sigmoid(x)
    return grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))


@register_decomposition([aten.embedding_dense_backward.default])
def embedding_dense_backward_decomposition(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
    """Decompose embedding_dense_backward using one-hot + tensordot."""
    embed_dim = grad_output.shape[-1]
    indices_flat = indices.reshape(-1).to(torch.int64)
    grad_flat = grad_output.reshape(-1, embed_dim)

    # Handle scale_grad_by_freq
    if scale_grad_by_freq:
        one_hot_counts = torch.nn.functional.one_hot(indices_flat, num_weights).to(grad_flat.dtype)
        counts = one_hot_counts.sum(dim=0)
        counts = torch.clamp(counts, min=1.0)
        scale = 1.0 / counts
        grad_flat = grad_flat * scale[indices_flat].unsqueeze(-1)

    # Zero out gradients for padding positions
    if padding_idx >= 0:
        mask = (indices_flat != padding_idx).unsqueeze(-1).to(grad_flat.dtype)
        grad_flat = grad_flat * mask

    # scatter-add
    indices_one_hot = torch.nn.functional.one_hot(indices_flat, num_weights).to(grad_flat.dtype)
    grad_weight = torch.tensordot(indices_one_hot, grad_flat, dims=([0], [0]))

    return grad_weight


@register_decomposition([aten.linear_backward])
def linear_backward_decomposition(input_, grad_output_, weight_, output_mask):
    """Decompose linear backward pass into matmul and tensordot.

    Linear forward: y = xA^T + b

    Args:
        input_ (torch.Tensor): Input tensor (x).
        grad_output_ (torch.Tensor): Output gradient (dy).
        weight_ (torch.Tensor): Weight matrix (A).
        output_mask (list[bool]): Which gradients to compute:
            [0] input gradient, [1] weight gradient, [2] bias gradient.

    Returns:
        tuple: (grad_input, grad_weight, grad_bias), None for masked outputs.
    """
    grad_input = None
    grad_weight = None
    grad_bias = None
    if output_mask[0]:
        grad_input = torch.matmul(grad_output_, weight_)
    if output_mask[1] or output_mask[2]:
        # Must compute both grad_weight and grad_bias together to match PyTorch meta registration
        batch_dims = "".join(chr(ord("a") + j) for j in range(input_.ndim - 1))
        einsum_eq = f"{batch_dims}i,{batch_dims}j->ij"
        grad_weight = torch.einsum(einsum_eq, grad_output_, input_)
        if grad_output_.ndim == 1:
            grad_bias = grad_output_.clone()
        else:
            grad_bias = torch.sum(grad_output_, dim=tuple(range(grad_output_.ndim - 1)))
    return (grad_input, grad_weight, grad_bias)


@register_decomposition([aten._foreach_pow.Scalar])
def _foreach_pow_scalar(tensors, exponent):
    """Decompose _foreach_pow.Scalar into individual pow operations"""
    return [torch.pow(t, exponent) for t in tensors]


@register_decomposition([aten.__rshift__.Scalar])
def rshift_scalar_decomp(x, shift):
    """Decompose right shift into floor division by power of 2"""
    divisor = 2**shift
    return torch.div(x, divisor, rounding_mode="floor").to(x.dtype)


@register_decomposition([aten.as_strided_scatter.default])
def as_strided_scatter_decomp(input, src, size, stride, storage_offset=0):
    """Decompose as_strided_scatter to index_copy"""
    # Ensure contiguous to match clone() semantics
    input = input.contiguous()
    input_flat = input.flatten()
    src_flat = src.flatten()

    # Compute linear indices for strided view
    idx = input.new_zeros(size, dtype=torch.long)

    for dim, s in enumerate(size):
        arange = torch.arange(s, dtype=torch.long, device=input.device)
        view_shape = [1] * len(size)
        view_shape[dim] = -1
        idx = idx + arange.view(view_shape) * stride[dim]

    flat_indices = idx.flatten() + storage_offset

    result_flat = input_flat.index_copy(0, flat_indices, src_flat)
    return result_flat.reshape(input.shape)


@register_decomposition([aten._grouped_mm.default])
def grouped_mm_decomposition(a, b, offs):
    """Decompose grouped matmul for torch.compile/MLIR paths.

    2D x 2D: a (d1, t), b (t, d2), offs (g,) -> (g, d1, d2) - calls NKI kernel
    2D x 3D: a (t, d1), b (g, d1, d2), offs (g,) -> (t, d2) - decomposes to primitives
    """
    # 2D x 2D case - call NKI kernel
    if a.dim() == 2 and b.dim() == 2:
        from torch_neuronx.python_ops.nki_kernels.grouped_mm import grouped_mm_2d_2d_nki

        return grouped_mm_2d_2d_nki(a, b, offs)

    # 2D x 3D case - decompose to primitives
    if a.dim() == 2 and b.dim() == 3:
        from torch_neuronx.utils import get_gmm_align

        t, d1 = a.shape
        g, _, d2 = b.shape
        align = get_gmm_align()

        b_index = (offs[:, None] // align <= torch.arange(t // align, device=a.device)).sum(0)
        b_index = torch.clamp(b_index, 0, g - 1)
        a_batched = a.reshape(t // align, align, d1)
        b_batched = b[b_index]
        r_batched = a_batched @ b_batched
        return r_batched.reshape(t, d2)

    return NotImplemented


def _validate_convolution_inputs(
    input,
    weight,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """Common validation for convolution operations using PyTorch-style checks."""
    k = input.ndim
    weight_dim = weight.ndim
    dim = weight_dim - 2

    if dim <= 0:
        raise RuntimeError("weight should have at least three dimensions")

    if groups <= 0:
        raise RuntimeError("non-positive groups is not supported")

    if any(p < 0 for p in padding):
        raise RuntimeError("negative padding is not supported")

    if transposed and any(p < 0 for p in output_padding):
        raise RuntimeError("negative output_padding is not supported")

    if any(s <= 0 for s in stride):
        raise RuntimeError("non-positive stride is not supported")

    if any(d <= 0 for d in dilation):
        raise RuntimeError("dilation should be greater than zero")

    if weight_dim != k:
        raise RuntimeError(
            f"Expected {weight_dim}-dimensional input for {weight_dim}-dimensional weight "
            f"{list(weight.shape)}, but got {k}-dimensional input of size "
            f"{list(input.shape)} instead"
        )

    if weight.shape[0] < groups:
        raise RuntimeError(
            f"Given groups={groups}, expected weight to be at least {groups} "
            f"at dimension 0, but got weight of size {list(weight.shape)} instead"
        )

    if weight.shape[0] % groups != 0:
        raise RuntimeError(
            f"Given groups={groups}, expected weight to be divisible by {groups} "
            f"at dimension 0, but got weight of size {list(weight.shape)} instead"
        )

    if not transposed:
        expected_input_channels = weight.shape[1] * groups
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given groups={groups}, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )

        for i in range(2, k):
            input_size = input.shape[i] + 2 * padding[i - 2]
            kernel_size = dilation[i - 2] * (weight.shape[i] - 1) + 1
            if input_size < kernel_size:
                raise RuntimeError(
                    f"Calculated padded input size per channel: ({input_size}). "
                    f"Kernel size: ({kernel_size}). Kernel size can't be greater than "
                    f"actual input size"
                )
    else:
        expected_input_channels = weight.shape[0]
        if input.shape[1] != expected_input_channels:
            raise RuntimeError(
                f"Given transposed=True, weight of size {list(weight.shape)}, "
                f"expected input{list(input.shape)} to have {expected_input_channels} channels, "
                f"but got {input.shape[1]} channels instead"
            )


def _validate_convolution_backward_inputs(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """Validate inputs for convolution backward operation."""
    _validate_convolution_inputs(
        input, weight, stride, padding, dilation, transposed, output_padding, groups
    )
    if grad_output.ndim != input.ndim:
        raise RuntimeError(
            f"Expected input and grad_output to have the same number of dimensions, "
            f"got: {input.ndim} and {grad_output.ndim}"
        )


def _tuple(x: list | tuple | int, ndim: int):
    """Convert x to a tuple of length ndim."""
    if isinstance(x, int):
        return (x,) * ndim
    # Convert to tuple, defaulting empty to zeros
    t = tuple(int(v) for v in x) if len(x) else (0,) * ndim
    # If shorter than ndim, repeat the last element to fill
    if len(t) < ndim:
        return t + (t[-1],) * (ndim - len(t))
    return t


@register_decomposition([aten.convolution_backward])
def torch_convolution_backward_overrideable(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    **kwargs,
):
    dtype = input.dtype
    ndim = input.ndim - 2  # spatial dimensions (1, 2, or 3)

    stride = _tuple(stride, ndim)
    padding = _tuple(padding, ndim)
    dilation = _tuple(dilation, ndim)
    output_padding = _tuple(output_padding, ndim)

    _validate_convolution_backward_inputs(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )

    n, cin = input.shape[:2]
    cout = weight.shape[0] if not transposed else weight.shape[1]
    spatial_in = input.shape[2:]
    spatial_out = grad_output.shape[2:]
    kernel_size = weight.shape[2:]

    need_gi = bool(output_mask[0]) if len(output_mask) > 0 else True
    need_gw = bool(output_mask[1]) if len(output_mask) > 1 else True
    need_gb = bool(output_mask[2]) if len(output_mask) > 2 else True

    grad_input = grad_weight = grad_bias = None

    # Select conv functions based on ndim
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][ndim - 1]
    conv_transpose_fn = [
        F.conv_transpose1d,
        F.conv_transpose2d,
        F.conv_transpose3d,
    ][ndim - 1]

    # --- grad_input ---
    if need_gi:
        if not transposed:
            # Compute output_padding to match input spatial size
            # output_size = (input_size - 1) * stride - 2 * padding +
            # dilation * (kernel - 1) + 1 + output_padding
            grad_input_output_padding = tuple(
                max(
                    0,
                    spatial_in[i]
                    - (
                        (spatial_out[i] - 1) * stride[i]
                        - 2 * padding[i]
                        + dilation[i] * (kernel_size[i] - 1)
                        + 1
                    ),
                )
                for i in range(ndim)
            )
            grad_input = conv_transpose_fn(
                grad_output,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
                output_padding=grad_input_output_padding,
                groups=groups,
                dilation=dilation,
            ).to(dtype)
        else:
            # transposed conv: grad_input = conv(grad_output, weight)
            grad_input = conv_fn(
                grad_output,
                weight,
                bias=None,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ).to(dtype)

    # --- grad_bias ---
    if need_gb and bias_sizes is not None and len(bias_sizes) > 0:
        reduce_dims = (0, *tuple(range(2, input.ndim)))
        grad_bias = grad_output.sum(dim=reduce_dims).to(dtype)

    # --- grad_weight ---
    if need_gw:
        cin_per_group = cin // groups
        cout_per_group = cout // groups
        is_depthwise = groups == cin and cin == cout and cin_per_group == 1

        if not transposed:
            if is_depthwise:
                # Depthwise convolution: use grouped conv trick for efficiency
                # Reshape: [N, C, *spatial] -> [C, N, *spatial] -> [1, C*N, *spatial]
                x_big = input.transpose(0, 1).reshape(1, cin * n, *spatial_in)
                k_big = grad_output.transpose(0, 1).reshape(cin * n, 1, *spatial_out).contiguous()

                dw_big = conv_fn(
                    x_big, k_big, bias=None, stride=1, padding=padding, dilation=1, groups=cin * n
                )

                # Subsample by dilation and slice to kernel_size
                if ndim == 1:
                    dw_big = dw_big[:, :, :: dilation[0]]
                    dw_big = dw_big[:, :, : kernel_size[0]]
                elif ndim == 2:
                    dw_big = dw_big[:, :, :: dilation[0], :: dilation[1]]
                    dw_big = dw_big[:, :, : kernel_size[0], : kernel_size[1]]
                else:
                    dw_big = dw_big[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                    dw_big = dw_big[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                # Reshape to [C, N, *K], sum over N -> [C, *K], then unsqueeze -> [C, 1, *K]
                dw_cnk = dw_big.reshape(cin, n, *kernel_size)
                grad_weight = dw_cnk.sum(dim=1).unsqueeze(1).to(dtype)
            else:
                # General grouped convolution
                dw_groups = []
                for g in range(groups):
                    x_g = input[:, g * cin_per_group : (g + 1) * cin_per_group]
                    dy_g = grad_output[:, g * cout_per_group : (g + 1) * cout_per_group]

                    x_t = x_g.transpose(0, 1).contiguous()
                    dy_t = dy_g.transpose(0, 1).contiguous()

                    dw_g = conv_fn(x_t, dy_t, bias=None, stride=1, padding=padding, dilation=stride)

                    # Subsample by dilation and slice to kernel_size
                    if ndim == 1:
                        dw_g = dw_g[:, :, :: dilation[0]]
                        dw_g = dw_g[:, :, : kernel_size[0]]
                    elif ndim == 2:
                        dw_g = dw_g[:, :, :: dilation[0], :: dilation[1]]
                        dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1]]
                    else:
                        dw_g = dw_g[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                        dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                    dw_groups.append(dw_g.transpose(0, 1))

                grad_weight = torch.cat(dw_groups, dim=0).to(dtype)
        else:
            # Transposed conv: weight is [Cin, Cout/groups, *kernel_size]
            dw_groups = []
            for g in range(groups):
                x_g = input[:, g * cin_per_group : (g + 1) * cin_per_group]
                dy_g = grad_output[:, g * cout_per_group : (g + 1) * cout_per_group]

                x_t = x_g.transpose(0, 1).contiguous()
                dy_t = dy_g.transpose(0, 1).contiguous()

                dw_g = conv_fn(dy_t, x_t, bias=None, stride=1, padding=padding, dilation=stride)

                # Subsample by dilation and slice to kernel_size
                if ndim == 1:
                    dw_g = dw_g[:, :, :: dilation[0]]
                    dw_g = dw_g[:, :, : kernel_size[0]]
                elif ndim == 2:
                    dw_g = dw_g[:, :, :: dilation[0], :: dilation[1]]
                    dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1]]
                else:
                    dw_g = dw_g[:, :, :: dilation[0], :: dilation[1], :: dilation[2]]
                    dw_g = dw_g[:, :, : kernel_size[0], : kernel_size[1], : kernel_size[2]]

                dw_groups.append(dw_g.transpose(0, 1))

            grad_weight = torch.cat(dw_groups, dim=0).to(dtype)

    return grad_input, grad_weight, grad_bias


@register_decomposition([aten.value_selecting_reduction_backward.default])
def value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim):
    """Decompose value_selecting_reduction_backward into scatter.

    Args:
        grad (torch.Tensor): Upstream gradient (shape of reduced output).
        dim (int): Dimension along which the forward reduction was performed.
        indices (torch.Tensor): Indices of selected values from the forward pass.
        sizes (list[int]): Shape of the original input tensor before reduction.
        keepdim (bool): Whether the forward op preserved the reduced dimension.

    Returns:
        torch.Tensor: Gradient with respect to the original input.
    """
    if dim < 0:
        dim = len(sizes) + dim
    grad_input = torch.zeros(sizes, dtype=grad.dtype, device=grad.device)
    if not keepdim and len(sizes) > 0:
        grad = grad.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    return grad_input.scatter(dim, indices, grad)


@register_decomposition([aten.histc.default])
def histc_decomposition(input, bins=100, min=0, max=0):
    """Decompose histc into basic ops

    Matches PyTorch's histc behavior:
    - Values in [min, max] are binned
    - Values < min or > max are ignored
    - Values exactly at max go to the last bin
    """
    if max == min:
        min = input.min().item()
        max = input.max().item()

        # Handle edge case where all values are identical
        if max == min:
            # When all values are identical, PyTorch puts all counts in the middle bin
            hist = torch.zeros(bins, dtype=torch.float32, device=input.device)
            hist[bins // 2] = float(input.numel())
            return hist

    bin_width = (max - min) / bins

    # Calculate bin indices
    bin_indices_float = (input - min) / bin_width
    bin_indices = bin_indices_float.long()

    # Create mask for values in valid range [min, max]
    # This filters out values < min or > max
    in_range_mask = (input >= min) & (input <= max)

    # Clamp indices to valid range [0, bins-1]
    bin_indices = torch.clamp(bin_indices, 0, bins - 1)

    # Create ones tensor with same dtype as input, but zero out entries that are out of range
    ones = torch.ones_like(bin_indices, dtype=input.dtype)
    ones = torch.where(in_range_mask, ones, torch.zeros_like(ones))

    # Scatter add - histogram must match dtype of ones for scatter_add
    hist = torch.zeros(bins, dtype=input.dtype, device=input.device)
    result = torch.scatter_add(hist, 0, bin_indices.flatten(), ones.flatten())

    return result


@register_decomposition([aten.argsort.default, aten.argsort.stable])
def argsort_decomposition(input, dim=-1, descending=False, stable=False):
    """Decompose argsort to sort"""
    return torch.sort(input, dim=dim, descending=descending, stable=stable)[1]


def get_decomposition_table(decompose_all: bool = False) -> dict:
    """Get the decomposition table for the Neuron backend.

    Args:
        decompose_all (bool): Unused, kept for API compatibility.

    Returns:
        dict: Mapping of ATen ops to their decomposition functions.
    """
    return neuron_decompositions


def get_compile_decomposition_table() -> dict:
    """Decomposition table for torch.compile (adds compile-only decompositions).

    This is needed for situations where we want a decomposition specifically
    for the torch.compile path but not the eager path.
    """
    return {
        **neuron_decompositions,
        aten.index_put.default: index_put_decomposition,
    }
