"""Context Parallel ring-attention handler for the overrideable SDPA op on Neuron.

Wires `aten._scaled_dot_product_fused_attention_overrideable.default` into PyTorch's
DTensor `_custom_op_handlers` so that seq-sharded SDPA inside `context_parallel(...)`
runs ring attention instead of the local-per-shard fallback driven by Strategy 4 in
`register_sharding.py`.

Everything this module touches on the upstream side is private, and installation happens at
`import torch_neuronx` time, so a torch bump that relocates those internals must not take the
whole backend import down with it. Install therefore degrades to a logged no-op; CP itself is
unusable in that state (`context_parallel` lives in the same relocated module), but everything
else in torch_neuronx still imports and runs. See `_install_cp_overrideable_handler`.
"""

import logging

import torch
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

aten = torch.ops.aten

logger = logging.getLogger(__name__)

# Upstream's private ring-attention templates, resolved once by
# `_install_cp_overrideable_handler`. They stay None if the import fails; the handlers below are
# then never registered, and each wrapper re-checks before use.
_templated_ring_attention = None
_templated_ring_attention_backward = None


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
