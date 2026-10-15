"""Shared Math2 formulas for SDPA."""

import torch


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _validate_dropout(dropout_p: float) -> None:
    if dropout_p < 0.0 or dropout_p > 1.0:
        raise ValueError(f"dropout_p={dropout_p} is outside [0, 1]. Pass a value in [0, 1].")


def _apply_attention_mask(
    logits: torch.Tensor,
    attn_bias: torch.Tensor | None,
    is_causal: bool,
) -> torch.Tensor:
    if is_causal:
        query_length, key_length = logits.shape[-2:]
        keep = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=logits.device,
        ).tril()
        logits = logits.masked_fill(keep.logical_not(), float("-inf"))

    if attn_bias is not None and attn_bias.numel() > 0:
        logits = logits + attn_bias

    return logits


def _apply_dropout(
    values: torch.Tensor,
    dropout_mask: torch.Tensor | None,
    dropout_p: float,
) -> torch.Tensor:
    _validate_dropout(dropout_p)
    # A lowering path sends a 1-element placeholder when it has no mask. Treat that
    # placeholder as absent. If not, the zero mask makes every gradient zero.
    if dropout_p <= 0.0 or dropout_mask is None or dropout_mask.numel() <= 1:
        return values
    if dropout_p == 1.0:
        return values * dropout_mask
    return values * dropout_mask / (1.0 - dropout_p)


def compute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None,
    dropout_mask: torch.Tensor | None,
    scale: float,
    is_causal: bool,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the shared SDPA Math2 forward pass.

    Args:
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        attn_bias (torch.Tensor | None): Additive attention bias, or ``None``.
        dropout_mask (torch.Tensor | None): Dropout mask; one element means no mask.
        scale (float): Scale applied to attention scores.
        is_causal (bool): Whether to apply a causal mask.
        dropout_p (float): Dropout probability in the inclusive range ``[0, 1]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Attention output and float32 log-sum-exp.

    Raises:
        ValueError: If ``dropout_p`` is outside ``[0, 1]``.
    """
    orig_dtype = query.dtype
    acc_dtype = _accumulator_dtype(orig_dtype)

    logits = query.to(acc_dtype) @ key.to(acc_dtype).transpose(-2, -1)
    logits = _apply_attention_mask(logits * scale, attn_bias, is_causal)

    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    fully_masked = max_logits == float("-inf")
    safe_max = torch.where(fully_masked, 0.0, max_logits)
    exp_logits = torch.exp(logits - safe_max)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    lse = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    safe_sum = torch.where(fully_masked, 1.0, sum_exp)
    probs = _apply_dropout(exp_logits / safe_sum, dropout_mask, dropout_p)

    output = probs.to(orig_dtype) @ value
    return output, lse.to(torch.float32)


def compute_attention_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None,
    dropout_mask: torch.Tensor | None,
    out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    is_causal: bool,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute the shared SDPA Math2 backward pass.

    Args:
        grad_out (torch.Tensor): Gradient of the attention output.
        query (torch.Tensor): Forward query tensor.
        key (torch.Tensor): Forward key tensor.
        value (torch.Tensor): Forward value tensor.
        attn_bias (torch.Tensor | None): Forward attention bias, or ``None``.
        dropout_mask (torch.Tensor | None): Forward dropout mask; one element means no mask.
        out (torch.Tensor): Forward attention output.
        lse (torch.Tensor): Forward log-sum-exp values.
        scale (float): Scale applied to attention scores.
        is_causal (bool): Whether the forward pass applied a causal mask.
        dropout_p (float): Forward dropout probability in the inclusive range ``[0, 1]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: Query, key, and
            value gradients; bias is ``None`` when absent or a one-element placeholder.

    Raises:
        ValueError: If ``dropout_p`` is outside ``[0, 1]``.
    """
    orig_dtype = query.dtype
    acc_dtype = _accumulator_dtype(orig_dtype)

    grad_out = grad_out.to(acc_dtype)
    query = query.to(acc_dtype)
    key = key.to(acc_dtype)
    value = value.to(acc_dtype)
    out = out.to(acc_dtype)
    lse = lse.to(acc_dtype)
    has_attn_bias = attn_bias is not None and attn_bias.numel() > 1
    if has_attn_bias:
        attn_bias = attn_bias.to(acc_dtype)

    logits = query @ key.transpose(-2, -1) * scale
    logits = _apply_attention_mask(logits, attn_bias, is_causal)

    fully_masked = lse == float("-inf")
    safe_lse = torch.where(fully_masked, 0.0, lse)
    probs = torch.exp(logits - safe_lse.unsqueeze(-1))
    probs = torch.where(fully_masked.unsqueeze(-1), 0.0, probs)

    masked_probs = _apply_dropout(probs, dropout_mask, dropout_p)
    grad_v = masked_probs.transpose(-2, -1) @ grad_out

    grad_out_v = grad_out @ value.transpose(-2, -1)
    grad_out_mask = _apply_dropout(grad_out_v, dropout_mask, dropout_p)

    local_out = masked_probs @ value
    sum_reduction = (grad_out_mask * probs).sum(dim=-1, keepdim=True) + (
        grad_out * (out - local_out)
    ).sum(dim=-1, keepdim=True)
    grad_logits = probs * (grad_out_mask - sum_reduction)
    ds = grad_logits * scale

    grad_q = ds @ key
    grad_k = ds.transpose(-2, -1) @ query
    grad_attn_bias = grad_logits.to(orig_dtype) if has_attn_bias else None

    return (
        grad_q.to(orig_dtype),
        grad_k.to(orig_dtype),
        grad_v.to(orig_dtype),
        grad_attn_bias,
    )
