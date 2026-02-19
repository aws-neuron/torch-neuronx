"""MLIR implementation of scaled_dot_product_attention"""

import math

import torch

from ...base import AttentionOpImpl, ExecutionResult
from ..kernel import TorchMlirKernel


def generate_dropout_mask(
    dropout_mask_shape, dropout_p, device, philox_seed, philox_offset, backward=False
):
    if dropout_p < 0.0 or dropout_p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {dropout_p}")

    if dropout_p == 0.0:
        dropout_mask = None
    elif float(dropout_p) == 1.0:
        dropout_mask = torch.zeros(dropout_mask_shape, dtype=torch.bool, device=device)
    else:
        # Generate this on CPU and bring to neuron
        dropout_mask = torch.empty(dropout_mask_shape, dtype=torch.bool)
        keep_prob = 1.0 - float(dropout_p)
        generator = torch.Generator()
        generator.set_state(philox_seed)
        dropout_mask = torch.bernoulli(dropout_mask, keep_prob, generator=generator).to(device)
        if not backward:
            # set rng state to reproduce same dropout mask as running on CPU
            torch.set_rng_state(generator.get_state())

    return dropout_mask


def compute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    dropout_mask: torch.Tensor,
    scale: float,
    is_causal: bool,
    dropout_p: float,
    mixed_precision: bool,
):
    """
    attention compute: softmax(Q @ K^T / sqrt(d)) @ V
    """
    orig_dtype = query.dtype
    acc_dtype = torch.float32 if mixed_precision else orig_dtype

    seq_len_q, seq_len_k = query.size(-2), key.size(-2)

    logits = query @ key.transpose(-2, -1)
    logits = logits.to(acc_dtype)
    logits = logits * scale

    if is_causal:
        temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        logits = logits + torch.where(temp_mask, 0.0, float("-inf"))
    elif attn_bias.numel() > 0:
        logits = logits + attn_bias

    # torch.logsumexp has MLIR lowering issues
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    lse = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    # softmax
    probs = exp_logits / sum_exp

    if dropout_p > 0.0 and dropout_mask.numel() > 0:
        if dropout_p == 1.0:
            probs = probs * dropout_mask
        else:
            probs = probs * dropout_mask / (1.0 - dropout_p)

    probs = probs.to(orig_dtype)
    output = probs @ value

    return output, lse.to(torch.float32)


class ScaledDotProductAttnMLIRImpl(AttentionOpImpl):
    """Native torch attention decomposed"""

    def __init__(self, op_name: str):
        self.kernel = TorchMlirKernel(
            compute_attention,
            op_name,
            static_argnums=(5, 6, 7, 8),  # scale, is_causal, dropout_p, mixed_precision
        )
        self.op_name = op_name

    def can_handle(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: float | None = None,
    ) -> bool:
        """Check if MLIR implementation can handle inputs"""
        if not super().can_handle(
            query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale
        ):
            return False

        # Check tensors are on neuron device
        if (
            query.device.type != "neuron"
            or key.device.type != "neuron"
            or value.device.type != "neuron"
        ):
            return False

        # Check dtype
        if not (query.dtype == key.dtype == value.dtype):
            return False

        # Check 4D tensors
        return not (query.ndim != 4 or key.ndim != 4 or value.ndim != 4)

    def _execute_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        scale: float | None = None,
    ) -> ExecutionResult:
        try:
            batch_size, q_heads, seq_len_q, head_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape

            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            # GQA
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            philox_seed = torch.get_rng_state()
            philox_offset = torch.empty((0,), dtype=torch.int32, device=query.device)

            # dropout mask
            dropout_mask_shape = (batch_size, q_heads, seq_len_q, seq_len_k)
            dropout_mask = generate_dropout_mask(
                dropout_mask_shape, dropout_p, query.device, philox_seed, philox_offset
            )

            # lowering issues when inputs to kernel were None
            if attn_bias is None:
                attn_bias = torch.zeros((1,), dtype=query.dtype, device=query.device)
            if dropout_mask is None:
                dropout_mask = torch.zeros((1,), dtype=query.dtype, device=query.device)

            attn_output, lse = self.kernel(
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                scale,
                is_causal,
                dropout_p,
                True,  # mixed_precision
            )

            result = (
                attn_output,
                lse,
                None,  # cum_seq_q
                None,  # cum_seq_k
                seq_len_q,
                seq_len_k,
                philox_seed,
                philox_offset,
                None,  # debug_attn_mask
            )
            return ExecutionResult(success=True, output=result)

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))


def compute_attention_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    dropout_mask: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    is_causal: bool,
    dropout_p: float,
    mixed_precision: bool,
):
    orig_dtype = query.dtype
    acc_dtype = torch.float32 if mixed_precision else orig_dtype

    grad_out = grad_out.to(acc_dtype)
    query = query.to(acc_dtype)
    key = key.to(acc_dtype)
    value = value.to(acc_dtype)
    lse = lse.to(acc_dtype)
    if attn_bias is not None:
        attn_bias = attn_bias.to(acc_dtype)

    logits = query @ key.transpose(-2, -1) * scale

    if is_causal:
        seq_len_q, seq_len_k = logits.shape[-2:]
        temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        logits = logits.masked_fill(temp_mask.logical_not(), float("-inf"))
    elif attn_bias is not None:
        logits = logits + attn_bias

    probs = torch.exp(logits - lse.unsqueeze(-1))

    if dropout_p > 0.0 and dropout_mask.numel() > 1:
        if dropout_p == 1.0:
            probs = probs * dropout_mask
        else:
            probs = probs * dropout_mask / (1.0 - dropout_p)

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

    grad_attn_bias = ds.to(orig_dtype) if attn_bias is not None else None

    grad_q = grad_q.to(orig_dtype)
    grad_k = grad_k.to(orig_dtype)
    grad_v = grad_v.to(orig_dtype)

    return grad_q, grad_k, grad_v, grad_attn_bias


class ScaledDotProductAttnBackwardMLIRImpl(AttentionOpImpl):
    """Attention backward compute"""

    def __init__(self, op_name: str):
        self.kernel = TorchMlirKernel(
            compute_attention_backward,
            op_name,
            static_argnums=(7, 8, 9, 10),  # scale, is_causal, dropout_p, mixed_precision
        )
        self.op_name = op_name

    def can_handle(
        self,
        grad_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
        grad_input_mask: list[bool],
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        cum_seq_q: torch.Tensor,
        cum_seq_k: torch.Tensor,
        max_q: int,
        max_k: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        philox_seed: torch.Tensor | None = None,
        philox_offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> bool:
        """Check if MLIR backward can handle inputs."""
        if not super().can_handle(query, key, value):
            return False

        # Check tensors are on neuron device
        tensors = [grad_out, out, logsumexp]
        if not all(t.device.type == "neuron" for t in tensors):
            return False

        # Check same dtype
        if not (query.dtype == grad_out.dtype == out.dtype):
            return False

        # Check 4D tensors
        if grad_out.ndim != 4 or out.ndim != 4:
            return False

        # Check shapes match
        batch_size, q_heads, seq_len_q, _ = query.shape
        batch_go, go_heads, seq_len_go, _ = grad_out.shape
        if seq_len_q != seq_len_go or q_heads != go_heads or batch_size != batch_go:
            return False

        # LSE shape validation
        return logsumexp.ndim == 3 and logsumexp.shape == (batch_size, q_heads, seq_len_q)

    def _execute_impl(
        self,
        grad_out: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor,
        grad_input_mask: list[bool],
        out: torch.Tensor,
        logsumexp: torch.Tensor,
        cum_seq_q: torch.Tensor,
        cum_seq_k: torch.Tensor,
        max_q: int,
        max_k: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        philox_seed: torch.Tensor | None = None,
        philox_offset: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> ExecutionResult:
        try:
            batch_size, q_heads, seq_len_q, head_dim = query.shape
            _, kv_heads, seq_len_k, _ = key.shape

            if scale is None:
                scale = 1.0 / math.sqrt(head_dim)

            # GQA
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            # dropout mask
            dropout_mask_shape = (batch_size, q_heads, seq_len_q, seq_len_k)
            dropout_mask = generate_dropout_mask(
                dropout_mask_shape,
                dropout_p,
                query.device,
                philox_seed,
                philox_offset,
                backward=True,
            )

            if dropout_mask is None:
                dropout_mask = torch.zeros((1,), dtype=query.dtype, device=query.device)

            grad_q, grad_k, grad_v, grad_attn_bias = self.kernel(
                grad_out,
                query,
                key,
                value,
                attn_bias,
                dropout_mask,
                logsumexp,
                scale,
                is_causal,
                dropout_p,
                True,  # mixed_precision
            )

            # Reduce GQA gradients
            if q_heads != kv_heads:
                repeat_factor = q_heads // kv_heads
                grad_k = grad_k.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim).sum(
                    dim=2
                )
                grad_v = grad_v.view(batch_size, kv_heads, repeat_factor, seq_len_k, head_dim).sum(
                    dim=2
                )

            return ExecutionResult(success=True, output=(grad_q, grad_k, grad_v, grad_attn_bias))

        except Exception as e:
            return ExecutionResult(success=False, error_msg=str(e))
