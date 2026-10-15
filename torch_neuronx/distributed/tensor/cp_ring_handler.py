"""Context Parallel ring-attention handler for the overrideable SDPA op on Neuron.

Wires `aten._scaled_dot_product_fused_attention_overrideable.default` into PyTorch's
DTensor `_custom_op_handlers` so that seq-sharded SDPA inside `context_parallel(...)`
runs ring attention instead of the local-per-shard fallback driven by Strategy 4 in
`register_sharding.py`.

Two forward implementations are wired in, selected per-call by `_use_ring_kernel`:
  - `_scaled_dot_product_ring_overrideable_attention`: PyTorch's `_templated_ring_attention`,
    which drives the ring in Python/c10d with N per-block dispatches to the overrideable op.
  - `_scaled_dot_product_monolithic_ring_fwd`: a single call to the fused NKI kernel
    `ring_attention_spmd_fwd`, which drives the entire ring (collectives + attention)
    inside one @nki.jit kernel. This is the default when the kernel's envelope allows it;
    `TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION=0` forces the templated ring, and every
    fallback logs why.

Everything this module touches on the upstream side is private, and installation happens at
`import torch_neuronx` time, so a torch bump that relocates those internals must not take the
whole backend import down with it. Install therefore degrades to a logged no-op; CP itself is
unusable in that state (`context_parallel` lives in the same relocated module), but everything
else in torch_neuronx still imports and runs. See `_install_cp_overrideable_handler`.
"""

import logging
import math

import torch
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torch_neuronx.neuron_dynamo_backend.settings import _getenv_bool

aten = torch.ops.aten

logger = logging.getLogger(__name__)

# Upstream's private ring-attention templates, resolved once by
# `_install_cp_overrideable_handler`. They stay None if the import fails; the handlers below are
# then never registered, and each wrapper re-checks before use.
_templated_ring_attention = None
_templated_ring_attention_backward = None

_FALLBACK = (
    "Fused NKI ring-attention kernel disabled: %s Falling back to _templated_ring_attention."
)

# Fallback reasons already logged. The gate runs on every layer's forward and backward,
# so warning unconditionally would repeat the same line hundreds of times per step.
_warned_fallback_reasons: set[str] = set()


def _warn_fallback_once(reason: str, detail: str) -> None:
    if reason not in _warned_fallback_reasons:
        _warned_fallback_reasons.add(reason)
        logger.warning(_FALLBACK, detail)


def _use_ring_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    num_workers: int,
    scale: float | None,
) -> bool:
    """Gate for the monolithic fused-NKI ring kernel.

    Enabled by default; `TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION=0` forces the templated
    ring. Every other check stands in for a kernel assert: when one rejects, log why
    (once per reason) and fall back.
    """
    from nkilib.core.attention.attention_cte import (
        _MAX_BS,
        _MAX_GLOBAL_CP_DEGREE,
        _Q_GRP_SZ,
    )
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        _cp_options,
    )

    if not _getenv_bool("TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION", True):
        _warn_fallback_once("env", "TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION is set to 0.")
        return False

    if attn_bias is not None:
        _warn_fallback_once("attn_bias", "the kernel has no attn_bias input.")
        return False
    if dropout_p != 0:
        _warn_fallback_once("dropout", f"the kernel has no dropout support, got {dropout_p=}.")
        return False
    # Both ring kernels resolve `softmax_scale or 1/sqrt(head_dim)`, so an explicit
    # scale=0.0 would silently become the default scale.
    if scale == 0.0:
        _warn_fallback_once("scale_zero", "the kernel cannot express scale=0.0.")
        return False

    if _cp_options.enable_load_balance:
        _warn_fallback_once(
            "load_balance",
            "the kernel assumes each rank holds a contiguous sequence slice, but the "
            "head-tail load balancer interleaves head+tail chunks, which would "
            "silently compute a wrong causal mask. Pass "
            "--parallelism.context_parallel_load_balancer=None (torchtitan) to use "
            "the kernel.",
        )
        return False

    b, q_h, seqlen, d = query.shape
    k_h = key.shape[1]

    # SDPA accepts empty tensors. The checks below divide by k_h and the adapters index
    # query.flatten()[0] and divide by sqrt(d), so reject before any shape arithmetic.
    if 0 in (b, q_h, k_h, seqlen, d):
        _warn_fallback_once(
            "empty",
            "the kernel requires non-empty inputs, got "
            f"query {tuple(query.shape)} vs key {tuple(key.shape)}.",
        )
        return False
    if key.shape != value.shape:
        _warn_fallback_once(
            "kv_shape",
            f"the kernel requires key.shape == value.shape, got {tuple(key.shape)} "
            f"vs {tuple(value.shape)}.",
        )
        return False
    # The kernel reshapes K/V using Q's batch, seqlen and head_dim, so K/V that differ
    # in any of them (e.g. cross-attention with its own K/V sequence length) would be
    # misread rather than rejected.
    if (key.shape[0], key.shape[2], key.shape[3]) != (b, seqlen, d):
        _warn_fallback_once(
            "qkv_shape",
            "the kernel requires K/V to match Q in batch, seqlen and head_dim, got "
            f"query {tuple(query.shape)} vs key {tuple(key.shape)}.",
        )
        return False
    if seqlen % _Q_GRP_SZ != 0:
        _warn_fallback_once(
            "seqlen", f"per-rank seqlen must be divisible by {_Q_GRP_SZ}, got {seqlen}."
        )
        return False
    if d > 128:
        _warn_fallback_once("head_dim", f"the kernel requires head_dim <= 128, got {d}.")
        return False
    if q_h % k_h != 0:
        _warn_fallback_once(
            "gqa_heads",
            f"q_heads must be a multiple of kv_heads to broadcast K/V, got {q_h=}, {k_h=}.",
        )
        return False
    # attention_cte bounds batch*heads after the GQA broadcast, hence q_h not k_h.
    if b * q_h > _MAX_BS:
        _warn_fallback_once(
            "max_bs", f"attention_cte is untested above batch*heads={_MAX_BS}, got {b * q_h}."
        )
        return False
    # global_cp_deg only reaches attention_cte on the causal path.
    if is_causal and num_workers > _MAX_GLOBAL_CP_DEGREE:
        _warn_fallback_once(
            "cp_degree",
            f"attention_cte is untested above global_cp_deg={_MAX_GLOBAL_CP_DEGREE}, "
            f"got {num_workers}.",
        )
        return False
    return True


def _broadcast_kv_for_gqa(
    key: torch.Tensor, value: torch.Tensor, q_h: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat K/V heads up to q_h for GQA, matching decompositions.py's GQA handling.

    ring_attention_spmd_fwd/bwd assert q_h == k_h internally. A no-op when they match.
    """
    k_h = key.shape[1]
    if k_h == q_h:
        return key, value
    repeat_factor = q_h // k_h
    return key.repeat_interleave(repeat_factor, dim=1), value.repeat_interleave(
        repeat_factor, dim=1
    )


def _reduce_kv_grad_for_gqa(
    grad_key: torch.Tensor, grad_value: torch.Tensor, k_h: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum GQA-broadcast K/V grads back down to the original k_h heads.

    Inverse of _broadcast_kv_for_gqa; a no-op when the grads are already at k_h heads.
    """
    q_h = grad_key.shape[1]
    if q_h == k_h:
        return grad_key, grad_value
    repeat_factor = q_h // k_h
    b, _, d, s = grad_key.shape
    grad_key = grad_key.view(b, k_h, repeat_factor, d, s).sum(dim=2)
    grad_value = grad_value.view(b, k_h, repeat_factor, d, s).sum(dim=2)
    return grad_key, grad_value


def _scaled_dot_product_monolithic_ring_fwd(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor, ...]:
    """Forward ring attention via a single call to the fused NKI kernel.

    query/key/value are the per-rank local shards (already at Shard(2)); the kernel
    circulates K/V across the CP ranks internally, so the whole ring runs inside one
    @nki.jit call instead of `_templated_ring_attention`'s N per-block dispatches.
    """
    import torch.distributed as dist
    from nkilib.experimental.attention.ring_attention_fwd import ring_attention_spmd_fwd

    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    q_h, L, E = query.shape[1], query.shape[-2], query.shape[-1]  # noqa: N806
    S = key.shape[2]  # noqa: N806
    key, value = _broadcast_kv_for_gqa(key, value, q_h)

    pg = mesh.get_group()
    num_workers = dist.get_world_size(pg)
    replica_groups = (tuple(dist.get_process_group_ranks(pg)),)

    if scale is None:
        scale = 1.0 / math.sqrt(E)

    # Causal contract: kernel requires scale=1.0 and a pre-scaled Q under causal
    # masking (attention_cte requirement); non-causal passes scale through as-is.
    softmax_scale = scale
    q = query
    if is_causal:
        q = (query.float() * scale).to(query.dtype)
        softmax_scale = 1.0

    grid = (int(get_logical_neuron_cores()),)
    wrapped = wrap_nki(ring_attention_spmd_fwd)
    output, lse_tiled = wrapped[grid](
        q,
        key,
        value,
        replica_groups=replica_groups,
        num_workers=num_workers,
        softmax_scale=softmax_scale,
        use_causal_mask=is_causal,
        striped_input=False,
        training=True,
        tp_q=True,
        tp_k=True,
    )

    # Kernel LSE layout is tiled [B, H, 128, seqlen//128]; invert to the natural
    # [B, H, L] layout the rest of the CP/backward machinery expects.
    b, h = lse_tiled.shape[:2]
    logsumexp = lse_tiled.transpose(-2, -1).reshape(b, h, -1).to(torch.float32)

    # Synthesize the remaining outputs of the overrideable-op 9-tuple to match
    # sdpa_fused_overrideable's no-dropout convention.
    zero = (query.flatten()[0] * 0).to(torch.int32)
    return (
        output,
        logsumexp,
        None,  # cum_seq_q
        None,  # cum_seq_k
        L,  # max_q
        S,  # max_k
        zero,  # philox_seed
        zero,  # philox_offset
        None,  # debug_mask
    )


def _scaled_dot_product_ring_overrideable_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    # Match upstream CP wrappers: _scaled_dot_product_ring_flash_attention raises the same.
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")
    # Match upstream CP wrappers: _scaled_dot_product_ring_efficient_attention raises the same.
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")

    num_workers = torch.distributed.get_world_size(mesh.get_group())
    if _use_ring_kernel(query, key, value, attn_bias, dropout_p, is_causal, num_workers, scale):
        return _scaled_dot_product_monolithic_ring_fwd(mesh, query, key, value, is_causal, scale)

    if _templated_ring_attention is None:
        raise RuntimeError(
            "Context Parallel: _templated_ring_attention is not available, "
            "likely an import issue."
        )
    return _templated_ring_attention(
        mesh.get_group(),
        seq_dim=2,
        op=aten._scaled_dot_product_fused_attention_overrideable.default,
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
    )


def _scaled_dot_product_monolithic_ring_bwd(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Backward ring attention via a single call to the fused NKI kernel.

    Mirrors _scaled_dot_product_monolithic_ring_fwd, with the ring roles transposed:
    the kernel circulates Q/dY/LSE across the CP ranks and keeps K/V local.

    Unlike the forward kernel, ring_attention_spmd_bwd applies `softmax_scale` as a
    plain post-multiply with no scale=1.0/pre-scaled-Q contract (that constraint is
    specific to attention_cte, which only the forward kernel calls), so `scale` is
    passed straight through here.
    """
    from nkilib.experimental.attention.ring_attention_bwd import ring_attention_spmd_bwd

    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    q_h, E = query.shape[1], query.shape[-1]  # noqa: N806
    k_h = key.shape[1]
    key, value = _broadcast_kv_for_gqa(key, value, q_h)
    if scale is None:
        scale = 1.0 / math.sqrt(E)

    pg = mesh.get_group()
    num_workers = torch.distributed.get_world_size(pg)
    replica_groups = (tuple(torch.distributed.get_process_group_ranks(pg)),)

    # Kernel wants Q/K/V/O/dY as [B, N, D, S].
    q_t = query.transpose(-2, -1).contiguous()
    k_t = key.transpose(-2, -1).contiguous()
    v_t = value.transpose(-2, -1).contiguous()
    o_t = out.transpose(-2, -1).contiguous()
    dy_t = grad_out.transpose(-2, -1).contiguous()
    # Convert natural [B, H, S] LSE to the kernel's tiled [B, H, 128, S//128]
    # layout (inverse of the forward adapter's LSE conversion).
    lse_b, lse_h, lse_s = logsumexp.shape
    lse_tiled = logsumexp.reshape(lse_b, lse_h, lse_s // 128, 128).transpose(-2, -1).contiguous()

    lnc_size = int(get_logical_neuron_cores())
    grid = (lnc_size,)
    wrapped = wrap_nki(ring_attention_spmd_bwd)
    dq_t, dk_t, dv_t = wrapped[grid](
        q_t,
        k_t,
        v_t,
        o_t,
        dy_t,
        lse_tiled,
        use_causal_mask=is_causal,
        mixed_precision=True,
        softmax_scale=scale,
        num_workers=num_workers,
        lnc_size=lnc_size,
        replica_groups=replica_groups,
        striped_attention=False,
    )

    grad_query = dq_t.transpose(-2, -1)
    grad_key = dk_t.transpose(-2, -1)
    grad_value = dv_t.transpose(-2, -1)
    grad_key, grad_value = _reduce_kv_grad_for_gqa(grad_key, grad_value, k_h)
    return grad_query, grad_key, grad_value, None


def _scaled_dot_product_ring_overrideable_attention_backward(
    mesh: DeviceMesh,
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
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    # An empty bias is no bias; normalizing keeps this gate's verdict identical to
    # the forward's, so fwd and bwd can never take different ring paths.
    bias = attn_bias if attn_bias is None or attn_bias.numel() != 0 else None
    num_workers = torch.distributed.get_world_size(mesh.get_group())
    if _use_ring_kernel(query, key, value, bias, dropout_p, is_causal, num_workers, scale):
        return _scaled_dot_product_monolithic_ring_bwd(
            mesh, grad_out, query, key, value, out, logsumexp, is_causal, scale
        )

    if _templated_ring_attention_backward is None:
        raise RuntimeError(
            "Context Parallel: _templated_ring_attention_backward is not available, "
            "likely an import issue."
        )
    return _templated_ring_attention_backward(
        mesh.get_group(),
        seq_dim=2,
        op=aten._scaled_dot_product_fused_attention_overrideable_backward.default,
        # named params consumed by _templated_ring_attention_backward itself
        grad_out=grad_out,
        grad_out_name="grad_out",
        query=query,
        key=key,
        value=value,
        out=out,
        logsumexp=logsumexp,
        is_causal=is_causal,
        # forwarded via **kwargs to the per-block op call
        attn_bias=attn_bias,
        grad_input_mask=grad_input_mask,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        scale=scale,
    )


_FWD_OP = aten._scaled_dot_product_fused_attention_overrideable.default
_BWD_OP = aten._scaled_dot_product_fused_attention_overrideable_backward.default


def _overrideable_handler(
    op_call: OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    # sharding_propagator will be removed by pytorch in the future, keep the code style uptodate
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    if output_sharding is None:
        raise RuntimeError(
            "CP overrideable handler: sharding propagation produced no output sharding for "
            f"{op_call}."
        )
    if output_sharding.needs_redistribute:
        raise RuntimeError(
            "CP overrideable handler expects inputs already at Shard(2); "
            "redistribution would defeat the ring algorithm."
        )

    if op_call is _FWD_OP:
        wrapper = _scaled_dot_product_ring_overrideable_attention
    elif op_call is _BWD_OP:
        wrapper = _scaled_dot_product_ring_overrideable_attention_backward
    else:
        raise NotImplementedError(f"unexpected op for CP overrideable handler: {op_call}")

    local = wrapper(
        op_info.compute_mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )
    return DTensor._op_dispatcher.wrap(local, output_sharding.output_spec)


def _install_cp_overrideable_handler() -> None:
    """Register the ring-attention handlers, or no-op if upstream internals moved.

    Failing here means CP is unavailable, not silently slower: `context_parallel` comes from the
    same private module, and the Strategy 4 seq-shard fallback in `register_sharding.py` has each
    rank attend only to its own chunk, which is wrong rather than merely degraded. The point of
    the no-op is to keep that breakage scoped to CP instead of failing `import torch_neuronx`.
    """
    global _templated_ring_attention, _templated_ring_attention_backward

    try:
        from torch.distributed.tensor.experimental._context_parallel import (
            _attention as _cp_attention,
        )
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            _templated_ring_attention as _fwd_template,
        )
        from torch.distributed.tensor.experimental._context_parallel._attention import (
            _templated_ring_attention_backward as _bwd_template,
        )

    except (ImportError, AttributeError):
        logger.warning(
            "Could not install the Neuron context-parallel ring-attention handler for "
            "aten._scaled_dot_product_fused_attention_overrideable: torch's private "
            "torch.distributed.tensor.experimental._context_parallel._attention internals are "
            "not where we expect them (torch %s). Context parallelism is unavailable; the rest "
            "of torch_neuronx is unaffected.",
            torch.__version__,
            exc_info=True,
        )
        return

    _templated_ring_attention = _fwd_template
    _templated_ring_attention_backward = _bwd_template
    _cp_attention.custom_ops[_FWD_OP] = _overrideable_handler
    _cp_attention.custom_ops[_BWD_OP] = _overrideable_handler


_install_cp_overrideable_handler()
