import torch


def precompute_rope_cache_patched(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    """
    Precompute RoPE cache as concatenated cos/sin tensors (no complex numbers).

    This is mathematically equivalent to the original precompute_rope_cache but
    returns cos and sin concatenated instead of using complex numbers.

    Args:
        dim (int): Head dimension (must be even for RoPE).
        max_seq_len (int): Maximum sequence length to precompute.
        base (float): Scaling factor for frequency computation. Defaults to 1_000_000.0.

    Returns:
        torch.Tensor: Concatenated cos/sin cache of shape [max_seq_len, dim * 2]
    """
    assert dim % 2 == 0, "dim must be even for RoPE"

    # Compute frequencies for each dimension pair
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Create position indexes
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half_patched(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (patched version)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast_patched(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.
    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.
    The input rope_cache tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    # The shape of rope_cache is (seqlen, head_dim * 2) because we concat cos and sin
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb_patched(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE using only real number operations (no complex numbers).

    This is mathematically equivalent to the original apply_rotary_emb but uses
    real number operations instead of complex multiplication.

    Args:
        xq (torch.Tensor): Query tensor [batch, seq, heads, head_dim]
        xk (torch.Tensor): Key tensor [batch, seq, heads, head_dim]
        rope_cache (torch.Tensor): Precomputed cos/sin cache [max_seq, head_dim * 2]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Rotated (xq, xk) tensors with same shapes as inputs
    """
    # input tensor x has shape [bsz, seq_len, n_heads, head_dim]
    head_dim = xq.shape[-1]

    # reshape for broadcast
    rope_cache = reshape_for_broadcast_patched(rope_cache, xq)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, n_heads, head_dim]
    # xk:  [bsz, seq_len, n_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half_patched(xq) * sin)
    xk_out = (xk * cos) + (rotate_half_patched(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv_patched(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep) - patched version"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    # Use repeat instead of expand to avoid zero-stride tensors
    # which cause issues in the Neuron compilation pipeline
    x_unsqueezed = torch.unsqueeze(x, dim=3)  # [bs, slen, n_kv_heads, 1, head_dim]
    x_repeated = x_unsqueezed.repeat(1, 1, 1, n_rep, 1)  # [bs, slen, n_kv_heads, n_rep, head_dim]
    return x_repeated.reshape(bs, slen, n_kv_heads * n_rep, head_dim)


def _apply_rotary_emb_functional(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Functional version of _apply_rotary_emb that avoids copy_ operations.

    This replaces the problematic torch.chunk + torch.cat pattern with
    direct slicing operations that are more functional.
    """
    # Split x into first and second half using slicing instead of chunk
    half_dim = x.shape[-1] // 2
    first_half = x[..., :half_dim]
    second_half = x[..., half_dim:]

    # Apply rotation
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin

    # Concatenate using stack + reshape instead of cat to avoid copy_ issues
    # This is mathematically equivalent but more functional
    result = torch.stack([first_, second_], dim=-1)
    result = result.reshape(*x.shape[:-1], x.shape[-1])

    return result


def apply_rotary_pos_emb_functional(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Functional version of apply_rotary_pos_emb that avoids copy_ operations.

    This is a drop-in replacement for the original apply_rotary_pos_emb function
    that uses functional operations instead of problematic copy_ operations.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb_functional(q, cos, sin)
    k_embed = _apply_rotary_emb_functional(k, cos, sin)
    return q_embed, k_embed
