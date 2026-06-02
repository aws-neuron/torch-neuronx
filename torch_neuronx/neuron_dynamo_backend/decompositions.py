# ruff: noqa: N806, E402
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
from nkilib.core.attention.attention_bwd import attention_bwd
from nkilib.core.attention.attention_cte import attention_cte
from torch._decomp import get_decompositions

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool
from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
    _has_nki_elementwise_kernels as _has_nki_foreach,
)
from torch_neuronx.python_ops.nki_kernels.rng_state import _has_nki_rng

if _has_nki_rng:
    from torch_neuronx.python_ops.nki_kernels.rng_state import rand

if _has_nki_foreach:
    from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
        P_SIZE as _FOREACH_P_SIZE,
    )
    from torch_neuronx.python_ops.nki_kernels.foreach_elementwise import (
        _add_scalar_kernel,
        _add_tensor_kernel,
        _div_scalar_kernel,
        _div_tensor_kernel,
        _mul_scalar_kernel,
        _mul_tensor_kernel,
        _sqrt_kernel,
        _sub_scalar_kernel,
        _sub_tensor_kernel,
    )

_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

logger = logging.getLogger(__name__)

wrapped_flash_fwd = wrap_nki(attention_cte)
wrapped_flash_bwd = wrap_nki(attention_bwd)

# Build the base decomposition table from standard PyTorch decompositions
# for ops that are not supported by the backend (yet)
neuron_decompositions = get_decompositions(
    [
        aten.logsumexp,
        aten.squeeze.dims,
        aten.binary_cross_entropy,
        aten.native_batch_norm,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten.native_batch_norm_backward,
        aten.native_group_norm_backward,
        aten.mse_loss_backward,
        aten.nansum,
        aten.addr,
        aten._unsafe_index,
        aten._unsafe_index_put,
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
        aten.softplus_backward,
        aten.col2im,  # accuracy mismatch on neuron, pending NCF-486
        aten.lerp,
        aten.elu,
        aten.elu_backward,
        aten.silu,
        aten.log_sigmoid_forward,
        aten.log_sigmoid_backward,
        aten.addmv,
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


@register_decomposition([aten.split.Tensor])
def split_decomposition(tensor, split_size, dim=0):
    """Decompose split into narrow/slice ops instead of as_strided.

    Args:
        tensor: Input tensor.
        split_size: Size of each chunk (int).
        dim: Dimension to split along.

    Returns:
        list[torch.Tensor]: List of split tensors.
    """
    if dim < 0:
        dim += tensor.ndim
    dim_size = tensor.shape[dim]
    chunks = []
    offset = 0
    while offset < dim_size:
        length = min(split_size, dim_size - offset)
        chunks.append(torch.ops.aten.slice.Tensor(tensor, dim, offset, offset + length))
        offset += length
    return chunks


@register_decomposition([aten.split_with_sizes.default])
def split_with_sizes_decomposition(tensor, split_sizes, dim=0):
    """Decompose split_with_sizes into narrow/slice ops instead of as_strided.

    Args:
        tensor: Input tensor.
        split_sizes: List of sizes for each chunk.
        dim: Dimension to split along.

    Returns:
        list[torch.Tensor]: List of split tensors.
    """
    if dim < 0:
        dim += tensor.ndim
    chunks = []
    offset = 0
    for length in split_sizes:
        chunks.append(torch.ops.aten.slice.Tensor(tensor, dim, offset, offset + length))
        offset += length
    return chunks


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
    # Match CPU behavior: topk does not support bool dtypes
    if x.dtype == torch.bool:
        raise RuntimeError("topk does not support bool dtypes on Neuron")

    # Handle 0-d scalar tensors: unsqueeze, run topk, then squeeze back
    if x.ndim == 0:
        values, indices = torch.topk(x.unsqueeze(0), k, dim=0, largest=largest, sorted=sorted)
        return values.squeeze(0), indices.squeeze(0)

    # AwsNeuronTopK only supports float types (float16, bfloat16, float32).
    # Cast integer types to float32, run topk, cast values back.
    if not x.dtype.is_floating_point:
        orig_dtype = x.dtype
        values, indices = torch.topk(
            x.to(torch.float32), k, dim=dim, largest=largest, sorted=sorted
        )
        return values.to(orig_dtype), indices

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


def _can_use_nki_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> bool:
    """Check if NKI flash attention kernel can be used.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.

    Returns:
        bool: True if NKI kernel constraints are satisfied.
    """
    if not _getenv_bool("TORCH_NEURONX_ENABLE_NKI_SDPA", True):
        return False

    B, H, L, E = query.shape
    S = key.shape[-2]
    D = key.shape[-1]

    # Conditions to use NKI SDPA kernel
    # NOTE: Limit for B * H needs to be set according to attention_cte kernel
    return (
        key.shape == value.shape
        and L % 512 == 0
        and S % 512 == 0
        and D <= 128
        and B * H <= 512
        and attn_bias is None
        and dropout_p == 0
    )


def _compute_attention_nki(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
    training: bool,
):
    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,)

    B, H, L, E = query.shape
    _, H_kv, S, D = key.shape
    q_nki = query.reshape(B * H, L, E)
    k_nki = key.reshape(B * H_kv, S, D)
    v_nki = value.reshape(B * H_kv, S, D)

    # attention_cte NKI kernel uses different default scale value
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    if training:
        output, neg_max, out_sum_recip = wrapped_flash_fwd[grid](
            q_nki,
            k_nki,
            v_nki,
            tp_q=True,
            tp_k=True,
            scale=scale,
            causal_mask=is_causal,
            cache_softmax=True,
        )

        # Compute LSE. NKI kernel produces tiled layout [B, H, B_P_SIZE, n_tile_q]
        # where lse[b,h,p,t] = LSE for sequence position t*B_P_SIZE + p.
        # Transpose to natural sequence order then flatten to ATen spec [B, H, S].
        logsumexp = -(neg_max + torch.log(out_sum_recip))
        logsumexp = logsumexp.to(torch.float32).reshape(B, H, *logsumexp.shape[-2:])
        logsumexp = logsumexp.transpose(-2, -1).reshape(B, H, -1)
        output = output.reshape(B, H, L, D)
        return output, logsumexp
    else:
        output = wrapped_flash_fwd[grid](
            q_nki,
            k_nki,
            v_nki,
            tp_q=True,
            tp_k=True,
            scale=scale,
            causal_mask=is_causal,
            cache_softmax=False,
        )
        output = output.reshape(B, H, L, D)
        return output


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
    _, Hkv, S, D = key.shape
    if Hkv < H and H % Hkv != 0:
        raise RuntimeError(
            "Number of heads in key and value must divide the number of heads in query"
        )

    if dropout_p > 0.0:
        philox_seed = torch.randint(0, 2**31 - 1, (1,), device=query.device, dtype=torch.int32)
        philox_offset = torch.zeros(1, device=query.device, dtype=torch.int32)
    else:
        philox_seed = (query.flatten()[0] * 0).to(torch.int32)
        philox_offset = (query.flatten()[0] * 0).to(torch.int32)

    if _can_use_nki_flash_attention(query, key, value, attn_bias, dropout_p):
        try:
            output, logsumexp = _compute_attention_nki(
                query, key, value, is_causal, scale, training=True
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
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> bool:
    """Check if NKI flash attention backward kernel can be used.

    Args:
        query (torch.Tensor): Query tensor of shape (B, H, L, E).
        key (torch.Tensor): Key tensor of shape (B, H, S, E).
        value (torch.Tensor): Value tensor of shape (B, H, S, E).
        attn_bias (torch.Tensor | None): Optional attention bias.
        dropout_p (float): Dropout probability.

    Returns:
        bool: True if NKI kernel constraints are satisfied.
    """
    if not _getenv_bool("TORCH_NEURONX_ENABLE_NKI_SDPA", True):
        return False

    if attn_bias is not None:
        return False

    _, _, L, E = query.shape
    _, _, S, _ = key.shape
    return (
        key.shape == value.shape
        and L % 512 == 0
        and S % 512 == 0
        and dropout_p == 0
        and E % math.ceil(E / 128) == 0
    )


def _compute_attention_backward_nki(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    scale: float | None,
):
    B, q_heads, L = query.shape[:3]
    kv_heads = key.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

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
            raise ValueError(f"Sequence length {L} must be divisible by tile size {tile_size}")
    else:
        lse_4d = logsumexp

    logical_neuron_cores = int(get_logical_neuron_cores())
    grid = (logical_neuron_cores,) if B * kv_heads % logical_neuron_cores == 0 else (1,)

    # Call NKI bwd kernel
    grad_q_nki, grad_k_nki, grad_v_nki = wrapped_flash_bwd[grid](
        q_nki,
        k_nki,
        v_nki,
        out_nki,
        grad_out_nki,
        lse_4d,
        use_causal_mask=is_causal,
        mixed_precision=True,
        softmax_scale=scale,
    )

    # Transform gradients back to ATen format
    grad_q = grad_q_nki.transpose(-2, -1)
    grad_k = grad_k_nki.transpose(-2, -1)
    grad_v = grad_v_nki.transpose(-2, -1)
    return grad_q, grad_k, grad_v


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
    can_use_nki = _can_use_nki_flash_attention_backward(query, key, value, attn_bias, dropout_p)
    # Try NKI
    if can_use_nki:
        try:
            grad_q, grad_k, grad_v = _compute_attention_backward_nki(
                grad_out,
                query,
                key,
                value,
                out,
                logsumexp,
                is_causal,
                scale,
            )

            return grad_q, grad_k, grad_v, None

        except Exception as e:
            logger.debug("SDPA backward NKI kernel call failed with exception:", e)
            pass

    # Fallback to manual backward decomposition
    orig_k_shape = key.shape

    # Prepare common inputs
    B, q_heads, L, E, S, orig_kv_heads, scale, is_gqa, key, value = _prepare_backward_inputs(
        query, key, value, orig_k_shape, scale
    )

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

    # 0-d (scalar) tensors: scatter requires ≥1-d, so handle directly.
    # CPU semantics: dim must be 0, index has one element (value 0),
    # source is 0-d → result is the source value with input's shape.
    if input.dim() == 0:
        return source.reshape(input.shape)

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

    Uses per-last-dim cumsum/scatter_add to keep reduce_window sizes small.
    For shape (..., C), reduce_window sizes are C and prod(shape[:-1]), not numel.
    No reshape needed — works directly on the original tensor shape.

    Args:
        tensor (torch.Tensor): Input tensor to find nonzero elements.
        size (int): Fixed output size. Pads with fill_value if fewer nonzeros.
        fill_value (int): Padding value for unused slots. Defaults to -1.
        with_count (bool): If True, return count of nonzero elements.

    Returns:
        tuple[torch.Tensor, int]: (indices, count) if with_count, else just indices.
    """
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

    shape = tensor.shape
    ndim = len(shape)
    C = shape[-1]
    device = tensor.device
    mask_f = mask.float()

    # --- Per-slice (along last dim): cumsum → scatter_add ---
    # reduce_window size = C

    # Within-slice rank of each element (1-based)
    rank = torch.cumsum(mask_f, dim=-1)

    # Scatter column index into per-slice output at rank position
    positions = torch.clamp(rank - 1, 0, C - 1).to(torch.long)
    col_src = torch.arange(C, device=device).float()
    col_values = torch.where(mask, col_src, 0.0)

    col_per_slice = torch.zeros_like(mask_f)
    col_per_slice = col_per_slice.scatter_add(-1, positions, col_values)
    # col_per_slice[..., k] = column of k-th nonzero in that slice

    # --- Global assembly ---
    # reduce_window size = prod(shape[:-1])

    slice_counts = mask_f.sum(dim=-1)  # shape[:-1]
    slice_offsets = torch.cumsum(slice_counts.flatten(), dim=0)  # [prod(shape[:-1])]
    offset_before = slice_offsets - slice_counts.flatten()

    # k-th slot in each slice → global position = offset_before + k
    k = torch.arange(C, device=device).float()
    global_pos = k + offset_before.reshape(shape[:-1] + (1,))
    valid = (k < slice_counts.unsqueeze(-1)) & (global_pos < size)
    global_pos = torch.clamp(global_pos, 0, size - 1).to(torch.long)

    # Scatter col coordinates into [size] output
    col_vals = torch.where(valid, col_per_slice, 0.0)
    out_col = torch.zeros(size, dtype=torch.float, device=device)
    out_col = out_col.scatter_add(0, global_pos.flatten(), col_vals.flatten())

    # Scatter leading-dim indices into [size] output
    # For N-d, the "row" is the flat index into shape[:-1]
    row_src = torch.arange(slice_counts.numel(), device=device).float()
    row_src = row_src.reshape(shape[:-1] + (1,)).expand_as(mask_f)
    row_vals = torch.where(valid, row_src, 0.0)
    out_row = torch.zeros(size, dtype=torch.float, device=device)
    out_row = out_row.scatter_add(0, global_pos.flatten(), row_vals.flatten())

    # Build coordinates
    if ndim == 1:
        coords = [out_col.to(torch.long)]
    else:
        # Unravel row index into leading dimensions
        row_flat = out_row.to(torch.long)
        leading = shape[:-1]
        coords = []
        for i, dim_size in enumerate(leading):
            stride = 1
            for d in leading[i + 1 :]:
                stride *= d
            coords.append((row_flat // stride) % dim_size)
        coords.append(out_col.to(torch.long))

    # Apply fill_value
    if fill_value is not None:
        fill_mask = torch.arange(size, device=device) >= num_nonzero
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

    Upcasts to float32 for intermediate computation to avoid precision loss
    in bf16, particularly for the (1 - sigmoid_x) term and the chain of multiplies.

    Args:
        grad_output (torch.Tensor): Gradient of the output.
        x (torch.Tensor): Input tensor from forward pass.

    Returns:
        torch.Tensor: Gradient with respect to input.
    """
    orig_dtype = grad_output.dtype
    grad_output = grad_output.to(torch.float32)
    x = x.to(torch.float32)
    sigmoid_x = torch.sigmoid(x)
    return (grad_output * sigmoid_x * (1 + x * (1 - sigmoid_x))).to(orig_dtype)


def _slice_backward_default(grad_output, input_sizes, dim, start, end, step):
    """Fallback equivalent to PyTorch's default slice_backward: zeros + slice_scatter."""
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)


@register_decomposition([aten.slice_backward.default])
def slice_backward_decomposition(grad_output, input_sizes, dim, start, end, step):
    """Decompose slice_backward to constant_pad_nd for the common step==1 case.

    PyTorch's default decomposition uses zeros + slice_scatter, which torch-mlir
    lowers to stablehlo.scatter (indirect indexed writes). For contiguous
    forward slices (step == 1), the gradient is just grad_output zero-padded
    along dim, which lowers to stablehlo.pad (memset + DMA copy). RoPE
    backward hits this path 4x per layer.

    Falls back to zeros + slice_scatter for step != 1 (rare, e.g. strided slices).
    """
    if step != 1:
        return _slice_backward_default(grad_output, input_sizes, dim, start, end, step)

    if dim < 0:
        dim = len(input_sizes) + dim

    full = input_sizes[dim]
    # Clamp to PyTorch slice semantics: start/end are clamped to [0, full].
    start_clamped = max(0, min(full, start if start >= 0 else full + start))
    end_clamped = max(0, min(full, end if end >= 0 else full + end))
    if end_clamped < start_clamped:
        end_clamped = start_clamped

    low = start_clamped
    high = full - end_clamped
    if low < 0 or high < 0:
        return _slice_backward_default(grad_output, input_sizes, dim, start, end, step)

    # constant_pad_nd takes pads in highest-dim-first pairs of (low, high).
    # Build a flat list with zeros for trailing dims after `dim`.
    rank = len(input_sizes)
    pad = []
    for d in range(rank - 1, dim - 1, -1):
        if d == dim:
            pad.extend([low, high])
        else:
            pad.extend([0, 0])
    return torch.ops.aten.constant_pad_nd.default(grad_output, pad, 0)


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
    """Decompose linear backward pass into 2D matmuls.

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
        # Must compute both grad_weight and grad_bias together to match PyTorch meta registration.
        # Flatten leading dims to a single batch dim and use a 2D mm so neuronx-cc sees a clean
        # reshape/transpose/dot_general instead of a unit-dim 4D transpose chain.
        n = grad_output_.shape[-1]
        k = input_.shape[-1]
        grad_output_2d = grad_output_.reshape(-1, n)
        input_2d = input_.reshape(-1, k)
        grad_weight = torch.mm(grad_output_2d.t(), input_2d)
        if grad_output_.ndim == 1:
            grad_bias = grad_output_.clone()
        else:
            grad_bias = torch.sum(grad_output_, dim=tuple(range(grad_output_.ndim - 1)))
    return (grad_input, grad_weight, grad_bias)


@register_decomposition([aten._foreach_pow.Scalar])
def _foreach_pow_scalar(tensors, exponent):
    """Decompose _foreach_pow.Scalar into individual pow operations"""
    return [torch.pow(t, exponent) for t in tensors]


@register_decomposition([aten._foreach_pow.ScalarAndTensor])
def _foreach_pow_scalar_and_tensor(scalar, exponents):
    """Decompose _foreach_pow.ScalarAndTensor (scalar ** tensor_list)"""
    return [torch.pow(scalar, e) for e in exponents]


@register_decomposition([aten._foreach_neg.default])
def _foreach_neg_decomp(self):
    return [torch.neg(t) for t in self]


@register_decomposition([aten._foreach_reciprocal.default])
def _foreach_reciprocal_decomp(self):
    return [torch.reciprocal(t) for t in self]


@register_decomposition([aten._foreach_abs.default])
def _foreach_abs_decomp(self):
    return [torch.abs(t) for t in self]


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


@register_decomposition([aten.searchsorted.Tensor])
def searchsorted_tensor_decomposition(
    sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None
):
    """Decompose searchsorted into comparison + sum.

    For each value, counts how many elements in sorted_sequence satisfy the
    boundary condition. This is O(N*M) rather than O(N*log M), but maps
    cleanly to StableHLO (no data-dependent control flow).

    Warning: Creates an intermediate [..., N, M] tensor from broadcasting
    values (N) against sorted_sequence (M). Large inputs may cause OOM.
    """
    if side is not None:
        right = side == "right"

    if sorter is not None:
        sorted_sequence = torch.gather(sorted_sequence, -1, sorter)

    # Broadcast: compare each value against all elements in sorted_sequence
    # sorted_sequence: [..., M], values: [..., N] -> compare [..., N, M]
    # NOTE: This creates an intermediate tensor of size batch * N * M.
    # For large M or N this may exceed device memory.
    seq_unsqueezed = sorted_sequence.unsqueeze(-2)  # [..., 1, M]
    val_unsqueezed = values.unsqueeze(-1)  # [..., N, 1]

    mask = seq_unsqueezed <= val_unsqueezed if right else seq_unsqueezed < val_unsqueezed

    result = mask.sum(dim=-1)

    if out_int32:
        return result.to(torch.int32)
    return result.to(torch.int64)


@register_decomposition([aten.searchsorted.Scalar])
def searchsorted_scalar_decomposition(
    sorted_sequence, value, *, out_int32=False, right=False, side=None, sorter=None
):
    """Decompose scalar searchsorted by wrapping the scalar as a tensor."""
    values = torch.tensor([value], dtype=sorted_sequence.dtype).to(sorted_sequence.device)
    result = searchsorted_tensor_decomposition(
        sorted_sequence, values, out_int32=out_int32, right=right, side=side, sorter=sorter
    )
    return result.squeeze(-1)


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
    bin_indices_unclamped = bin_indices_float.long()
    bin_indices = torch.clamp(bin_indices_unclamped, 0, bins - 1)

    # Create mask for values in valid range [min, max]
    # This filters out values < min or > max
    in_range_mask = (input >= min) & (input <= max)
    bin_range = torch.arange(bins, device=input.device, dtype=input.dtype)
    result_counts = torch.logical_and(
        in_range_mask.unsqueeze(0), bin_range.unsqueeze(1) == bin_indices.unsqueeze(0)
    )
    result = torch.sum(result_counts, dim=1).to(input.dtype)
    return result


@register_decomposition([aten.index_select_backward])
def index_select_backward_decomposition(grad_output, input_sizes, dim, index):
    """Decompose index_select_backward for torch.compile compatibility.

    This needs to be decomposed here otherwise will fail lowering in torch-mlir.
    """
    if dim < 0:
        dim = len(input_sizes) + dim

    grad_input = torch.zeros(input_sizes, dtype=grad_output.dtype, device=grad_output.device)

    shape = [1] * grad_output.ndim
    shape[dim] = -1
    index_expanded = index.view(shape).expand_as(grad_output)

    return grad_input.scatter_add(dim, index_expanded, grad_output)


@register_decomposition([aten.repeat_interleave.Tensor])
def repeat_interleave_tensor_decomposition(repeats, output_size=None):
    """Decompose repeat_interleave.Tensor for torch.compile compatibility.

    Computes output size upfront to avoid graph breaks from .item() calls.
    """
    if repeats.numel() == 0:
        return torch.empty_like(repeats)

    # Compute output size without graph break
    if output_size is None:
        output_size = repeats.sum()

    # Compute cumsum for indexing
    cumsum = repeats.cumsum(0)

    # Create result indices using arange and comparison
    indices = torch.arange(output_size, device=repeats.device)

    # For each position, find which repeat group it belongs to
    # by comparing against cumsum boundaries
    group_ids = (indices.unsqueeze(1) >= cumsum.unsqueeze(0)).sum(1)

    return group_ids


@register_decomposition([aten.select_scatter.default])
def select_scatter_decomposition(input, src, dim, index):
    """Normalize negative index for select_scatter.

    Converts negative index to positive, then delegates to the original op.
    Positive indices pass through without decomposition.

    Args:
        input (torch.Tensor): Input tensor.
        src (torch.Tensor): Source tensor to scatter.
        dim (int): Dimension along which to scatter.
        index (int): Index at which to scatter. Negative values are normalized.

    Returns:
        torch.Tensor or NotImplemented: Result with normalized index, or
            NotImplemented for non-negative indices.
    """
    if index >= 0 and dim >= 0:
        return NotImplemented
    if dim < 0:
        dim += input.ndim
    if index < 0:
        index += input.shape[dim]
    return torch.ops.aten.select_scatter.default(input, src, dim, index)


@register_decomposition([aten.argsort.default, aten.argsort.stable])
def argsort_decomposition(input, dim=-1, descending=False, stable=False):
    """Decompose argsort to sort"""
    return torch.sort(input, dim=dim, descending=descending, stable=stable)[1]


@register_decomposition([aten.native_dropout.default])
def native_dropout_decomposition(input, p: float, train: bool = True):
    """Decompose native_dropout using on-device RNG."""
    if train and not (0.0 <= float(p) <= 1.0):
        raise RuntimeError(f"dropout probability has to be between 0 and 1, but got {p}")

    if not train or p == 0.0:
        mask = torch.ones(input.shape, dtype=torch.bool, device=input.device)
        scale = 1.0
    elif p == 1.0:
        mask = torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        scale = 0.0
    else:
        keep_prob = 1.0 - p
        if _has_nki_rng:
            mask = rand(input, list(input.shape)) < keep_prob
        else:
            mask = (
                torch.rand(list(input.shape), dtype=torch.float32, device=input.device) < keep_prob
            )
        scale = 1.0 / keep_prob

    y = input * mask.to(input.dtype) * scale
    return y, mask


@register_decomposition([aten.addbmm.default])
def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    """Decompose addbmm: beta * self + alpha * (batch1 @ batch2).sum(dim=0)"""
    bmm_result = torch.bmm(batch1, batch2).sum(dim=0)
    if alpha != 1:
        bmm_result = bmm_result * alpha
    if beta == 0:
        return bmm_result
    if beta != 1:
        self = self * beta
    return self + bmm_result


@register_decomposition([aten.hypot.default])
def hypot(self, other):
    """Numerically stable hypot: max(|a|,|b|) * sqrt(1 + (min/max)^2)."""
    a = torch.abs(self)
    b = torch.abs(other)
    max_val = torch.maximum(a, b)
    min_val = torch.minimum(a, b)
    ratio = min_val / max_val
    return torch.where(max_val == 0, max_val, max_val * torch.sqrt(1 + ratio * ratio))


@register_decomposition([aten.batch_norm_stats.default])
def batch_norm_stats_decomposition(input, eps):
    """Compute batch normalization statistics (mean and inverse std)."""

    # Reduction axes: all except channel dimension (index 1)
    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)

    # Calculate mean and variance
    mean = torch.mean(input, dim=reduction_axes, keepdim=False)
    var = torch.var(input, dim=reduction_axes, unbiased=False, keepdim=False)

    # Calculate inverse standard deviation
    invstd = torch.rsqrt(var + eps)

    return mean, invstd


@register_decomposition([aten.batch_norm_elemt.default])
def batch_norm_elemt_decomposition(input, weight, bias, mean, invstd, eps):
    """Decompose batch_norm_elemt: (input - mean) * invstd * weight + bias."""
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.view(shape)
    invstd_reshaped = invstd.view(shape)
    weight_reshaped = weight.view(shape) if weight is not None else 1.0
    bias_reshaped = bias.view(shape) if bias is not None else 0.0
    normalized = (input - mean_reshaped) * invstd_reshaped
    return normalized * weight_reshaped + bias_reshaped


@register_decomposition([aten.batch_norm_backward_elemt.default])
def batch_norm_backward_elemt_decomposition(
    grad_output, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
):
    """Decompose batch_norm_backward_elemt into primitive ops."""
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.reshape(shape)
    invstd_reshaped = invstd.reshape(shape)
    weight_reshaped = weight.reshape(shape) if weight is not None else 1.0
    sum_dy_reshaped = sum_dy.reshape(shape)
    sum_dy_xmu_reshaped = sum_dy_xmu.reshape(shape)
    total_count = count.sum() if torch.is_tensor(count) else count
    xmu = input - mean_reshaped
    k = sum_dy_xmu_reshaped * invstd_reshaped * invstd_reshaped / total_count
    grad_input = (
        (grad_output - sum_dy_reshaped / total_count - xmu * k) * invstd_reshaped * weight_reshaped
    )
    return grad_input


@register_decomposition([aten.batch_norm_backward_reduce.default])
def batch_norm_backward_reduce_decomposition(
    grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g
):
    """Decompose batch_norm_backward_reduce into sum reductions."""
    # input_g, weight_g, bias_g are unused here because we unconditionally
    # return the grads - may be worth optimizing in the future

    reduction_axes = tuple(i for i in range(input.ndim) if i != 1)
    shape = [1] * input.ndim
    shape[1] = -1
    mean_reshaped = mean.view(shape)
    invstd_reshaped = invstd.view(shape)
    xmu = input - mean_reshaped
    xhat = xmu * invstd_reshaped
    sum_dy = torch.sum(grad_output, dim=reduction_axes, keepdim=False)
    sum_dy_xmu = torch.sum(grad_output * xmu, dim=reduction_axes, keepdim=False)
    grad_weight = torch.sum(grad_output * xhat, dim=reduction_axes, keepdim=False)
    grad_bias = sum_dy.clone()
    return sum_dy, sum_dy_xmu, grad_weight, grad_bias


@register_decomposition([aten.addcmul.default])
def addcmul_decomposition(input, tensor1, tensor2, value=1):
    """Decompose addcmul: input + value * tensor1 * tensor2."""
    return input + value * tensor1 * tensor2


@register_decomposition([aten.addcdiv.default])
def addcdiv_decomposition(input, tensor1, tensor2, value=1):
    """Decompose addcdiv: input + value * tensor1 / tensor2."""
    return input + value * tensor1 / tensor2


@register_decomposition([aten.erfinv.default])
def erfinv_decomposition(y):
    """Decompose erfinv using rational polynomial approximation."""
    if not y.is_floating_point():
        y = y.to(torch.float32)

    a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
    b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
    c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
    d = [3.543889200, 1.637067800]

    y_abs = torch.abs(y)
    result = torch.where(
        torch.isnan(y) | (y_abs > 1.0), y * float("nan"), torch.sign(y) * float("inf")
    )

    mask_valid = y_abs < 1.0
    z_low = y * y
    num_low = ((a[3] * z_low + a[2]) * z_low + a[1]) * z_low + a[0]
    dem_low = (((b[3] * z_low + b[2]) * z_low + b[1]) * z_low + b[0]) * z_low + 1.0
    x_low = y * num_low / dem_low

    z_high = torch.sqrt(-torch.log((1.0 - y_abs) / 2.0))
    num_high = ((c[3] * z_high + c[2]) * z_high + c[1]) * z_high + c[0]
    dem_high = (d[1] * z_high + d[0]) * z_high + 1.0
    x_high = torch.sign(y) * num_high / dem_high

    result = torch.where(mask_valid & (y_abs <= 0.7), x_low, result)
    result = torch.where(mask_valid & (y_abs > 0.7), x_high, result)

    return result


# ── foreach decompositions ───────────────────────────────────────────────────
# Decompose foreach ops into per-tensor NKI kernel calls with elementwise op
# fallback for small/empty/mismatched tensors.

_FOREACH_OPS = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}
_FOREACH_HAS_ALPHA = {"add", "sub"}
_FOREACH_TENSOR_KERNELS = {}
_FOREACH_SCALAR_KERNELS = {}
if _has_nki_foreach:
    _FOREACH_TENSOR_KERNELS = {
        "add": _add_tensor_kernel,
        "sub": _sub_tensor_kernel,
        "mul": _mul_tensor_kernel,
        "div": _div_tensor_kernel,
    }
    _FOREACH_SCALAR_KERNELS = {
        "add": _add_scalar_kernel,
        "sub": _sub_scalar_kernel,
        "mul": _mul_scalar_kernel,
        "div": _div_scalar_kernel,
    }


def _make_foreach_list_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_TENSOR_KERNELS.get(op)

    def decomp(self, other, alpha=1):
        result = []
        for a, b in zip(self, other, strict=False):
            if a.numel() == 0:
                result.append(a)
            elif _has_nki_foreach and a.numel() > 1 and a.shape == b.shape:
                out_dtype = torch.promote_types(a.dtype, b.dtype)
                if has_alpha:
                    alpha_t = torch.full(
                        (_FOREACH_P_SIZE, 1), alpha, dtype=out_dtype, device=a.device
                    )
                    result.append(
                        nki_kernel[(2,)](a.to(out_dtype), b.to(out_dtype), alpha_t, a.numel())
                    )
                else:
                    result.append(nki_kernel[(2,)](a.to(out_dtype), b.to(out_dtype), a.numel()))
            elif has_alpha:
                result.append(torch_op(a, b, alpha=alpha))
            else:
                result.append(torch_op(a, b))
        return result

    return decomp


def _make_foreach_scalar_decomp(op):
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_SCALAR_KERNELS.get(op)

    def decomp(self, scalar):
        result = []
        for t in self:
            compute_dtype = torch.result_type(t, scalar)
            if op == "div" and not compute_dtype.is_floating_point:
                compute_dtype = torch.float32
            if t.numel() == 0:
                result.append(torch.empty_like(t, dtype=compute_dtype))
            elif _has_nki_foreach and t.numel() > 1:
                scalar_t = torch.full(
                    (_FOREACH_P_SIZE, 1), scalar, dtype=compute_dtype, device=t.device
                )
                result.append(nki_kernel[(2,)](t.to(compute_dtype), scalar_t, t.numel()))
            else:
                result.append(torch_op(t.to(compute_dtype), scalar))
        return result

    return decomp


def _make_foreach_tensor_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]
    nki_kernel = _FOREACH_SCALAR_KERNELS.get(op)

    def decomp(self, other, alpha=1):
        effective = other * alpha if has_alpha else other
        result = []
        for t in self:
            if t.numel() == 0:
                result.append(t)
            elif _has_nki_foreach and t.numel() > 1:
                compute_dtype = torch.promote_types(t.dtype, effective.dtype)
                scalar_t = (
                    effective.to(compute_dtype)
                    .reshape(1, 1)
                    .expand(_FOREACH_P_SIZE, 1)
                    .contiguous()
                )
                result.append(nki_kernel[(2,)](t.to(compute_dtype), scalar_t, t.numel()))
            elif has_alpha:
                result.append(torch_op(t, other, alpha=alpha))
            else:
                result.append(torch_op(t, other))
        return result

    return decomp


def _make_foreach_list_inplace_decomp(op):
    has_alpha = op in _FOREACH_HAS_ALPHA
    torch_op = _FOREACH_OPS[op]

    if has_alpha:

        def decomp(self, other, alpha=1):
            for s, o in zip(self, other, strict=False):
                torch_op(s, o, alpha=alpha, out=s)
    else:

        def decomp(self, other):
            for s, o in zip(self, other, strict=False):
                torch_op(s, o, out=s)

    return decomp


def _make_foreach_scalar_inplace_decomp(op):
    torch_op = _FOREACH_OPS[op]

    def decomp(self, scalar):
        for s in self:
            torch_op(s, scalar, out=s)

    return decomp


for _op in ("add", "sub", "mul", "div"):
    register_decomposition([getattr(aten, f"_foreach_{_op}").List])(_make_foreach_list_decomp(_op))
    register_decomposition([getattr(aten, f"_foreach_{_op}").Scalar])(
        _make_foreach_scalar_decomp(_op)
    )
    register_decomposition([getattr(aten, f"_foreach_{_op}_").List])(
        _make_foreach_list_inplace_decomp(_op)
    )
    register_decomposition([getattr(aten, f"_foreach_{_op}_").Scalar])(
        _make_foreach_scalar_inplace_decomp(_op)
    )

for _op in ("add", "mul", "div"):
    register_decomposition([getattr(aten, f"_foreach_{_op}").Tensor])(
        _make_foreach_tensor_decomp(_op)
    )


@register_decomposition([aten._foreach_sqrt.default])
def _foreach_sqrt_decomp(self):
    result = []
    for t in self:
        if t.numel() == 0:
            result.append(t)
        elif _has_nki_foreach and t.numel() > 1:
            result.append(_sqrt_kernel[(2,)](t, t.numel()))
        else:
            result.append(torch.sqrt(t))
    return result


@register_decomposition([aten._foreach_sqrt_.default])
def _foreach_sqrt_inplace_decomp(self):
    for s in self:
        torch.sqrt(s, out=s)


@register_decomposition([aten._foreach_addcmul.Scalar])
def _foreach_addcmul_decomp(self, tensor1, tensor2, value=1):
    return [
        torch.addcmul(s, t1, t2, value=value)
        for s, t1, t2 in zip(self, tensor1, tensor2, strict=False)
    ]


@register_decomposition([aten._foreach_addcdiv.Scalar])
def _foreach_addcdiv_decomp(self, tensor1, tensor2, value=1):
    return [
        torch.addcdiv(s, t1, t2, value=value)
        for s, t1, t2 in zip(self, tensor1, tensor2, strict=False)
    ]


@register_decomposition([aten._foreach_addcmul_.Scalar])
def _foreach_addcmul_inplace_decomp(self, tensor1, tensor2, value=1):
    for s, t1, t2 in zip(self, tensor1, tensor2, strict=False):
        s.addcmul_(t1, t2, value=value)


@register_decomposition([aten._foreach_addcdiv_.Scalar])
def _foreach_addcdiv_inplace_decomp(self, tensor1, tensor2, value=1):
    for s, t1, t2 in zip(self, tensor1, tensor2, strict=False):
        s.addcdiv_(t1, t2, value=value)


def get_decomposition_table(decompose_all: bool = False) -> dict:
    """Get the decomposition table for the Neuron backend.

    Args:
        decompose_all (bool): Unused, kept for API compatibility.

    Returns:
        dict: Mapping of ATen ops to their decomposition functions.
    """
    return neuron_decompositions


def _cross_entropy_loss_decomp(
    input, target, weight=None, reduction=1, ignore_index=-100, label_smoothing=0.0
):
    """Decompose cross_entropy_loss into log_softmax + nll_loss for torch.compile.

    This prevents AOT autograd from dispatching to the neuron kernel's make_fx
    (which causes nested tracing conflicts). The decomposed ops are standard
    aten ops that the neuron backend compiles directly.

    Note: label_smoothing is not supported by this decomposition. When
    label_smoothing > 0, the decomposition is skipped and the op falls through
    to the neuron kernel (which handles it via the chunked CE implementation).
    """
    if label_smoothing != 0.0:
        # Fall through to the neuron kernel's implementation which supports
        # label smoothing via the chunked CE algorithm.
        return NotImplemented
    log_softmax = torch.log_softmax(input, dim=1)
    return torch.ops.aten.nll_loss_forward(log_softmax, target, weight, reduction, ignore_index)[0]


def get_compile_decomposition_table() -> dict:
    """Decomposition table for torch.compile (adds compile-only decompositions).

    During torch.compile, AOT autograd traces the joint forward+backward graph.
    Ops registered on PrivateUse1 (neuron kernels) would do nested make_fx inside
    AOT autograd's make_fx, causing FunctionalTensor/ProxyTensor mode conflicts.
    By adding these ops to the decomposition table, AOT autograd decomposes them
    into standard ops during tracing, avoiding the nesting issue entirely.

    These decompositions are intentionally NOT in the base neuron_decompositions
    table. In the eager path, cross_entropy_loss goes through the optimized chunked
    kernel (_ce_chunked_fwd_inner) which is memory-efficient at O(N * chunk_size).
    The naive log_softmax + nll_loss decomposition here is O(N * V) but is fine for
    torch.compile since the neuron backend compiles the full graph. Keeping these
    in the compile-only table preserves the eager path's optimized implementation.
    """
    compile_decomps = get_decompositions(
        [
            aten.nll_loss_forward,
        ]
    )
    compile_decomps[aten.cross_entropy_loss.default] = _cross_entropy_loss_decomp
    return {
        **neuron_decompositions,
        **compile_decomps,
    }
