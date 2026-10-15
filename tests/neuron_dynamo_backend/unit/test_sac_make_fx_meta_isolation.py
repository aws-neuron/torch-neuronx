"""Regression tests for an eager kernel trace inside a SAC region. CPU only.

A Neuron kernel traces its body with ``make_fx`` during eager execution. Inside a
selective activation checkpoint (SAC) region the SAC dispatch mode observes that
trace, stamps ``ac_graph_id`` into the global ``fx.traceback.current_meta`` and
records the traced ops in its storage. A later ``torch.compile`` then crashes in
``cleanup_recompute_tags``, and the backward recompute loses its storage entries.
These tests reproduce the torch mechanism. They make sure that
``_isolated_kernel_trace`` prevents it.
"""

import contextlib

import pytest
import torch
import torch.fx.traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from torch_neuronx.python_ops.torch_mlir.kernel import (
    TorchMlirKernel,
    _isolated_kernel_trace,
)


@pytest.fixture(autouse=True)
def _isolate_global_state():
    # The negative tests pollute the process-global current_meta on purpose.
    saved = dict(fx_traceback.current_meta)
    # A cached compiled function skips the partitioner and hides the crash.
    torch._dynamo.reset()
    yield
    fx_traceback.current_meta = saved


def _sac_policy(ctx, func, *args, **kwargs):
    # One MUST_SAVE op makes cleanup_recompute_tags compare ac_graph_id values.
    if func is torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _trace_kernel(a, *, isolate: bool):
    if isolate:
        with _isolated_kernel_trace():
            make_fx(lambda t: t * 2)(a.detach())
    else:
        make_fx(lambda t: t * 2)(a.detach())


def _run_checkpointed_compile(*, isolate: bool):
    """Compile inside a SAC region that traces a kernel first."""

    # aot_eager_default_partitioner runs cleanup_recompute_tags and returns a
    # boxed function.
    @torch.compile(backend="aot_eager_default_partitioner")
    def inner(x, w):
        return torch.relu(x @ w).sin()

    traced = []

    def block(a, b):
        # Trace once, like a lowering that the kernel cache keeps afterwards.
        if not traced:
            _trace_kernel(a, isolate=isolate)
            traced.append(1)
        return inner(a, b) + 1

    x = torch.randn(4, 4, requires_grad=True)
    w = torch.randn(4, 4, requires_grad=True)
    out = checkpoint(
        block,
        x,
        w,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy),
    )
    out.sum().backward()


def _run_checkpointed_eager(*, isolate: bool):
    """Run an eager SAC region that traces a kernel. Return the SAC storage."""
    fwd_ctx, bwd_ctx = create_selective_checkpoint_contexts(_sac_policy)

    def block(a, b):
        _trace_kernel(a, isolate=isolate)
        return (a @ b) + 1

    x = torch.randn(4, 4, requires_grad=True)
    w = torch.randn(4, 4, requires_grad=True)
    out = checkpoint(block, x, w, use_reentrant=False, context_fn=lambda: (fwd_ctx, bwd_ctx))
    out.sum().backward()
    return fwd_ctx.storage


def test_isolated_kernel_trace_leaves_current_meta_clean():
    saved = dict(fx_traceback.current_meta)
    _run_checkpointed_eager(isolate=True)
    assert dict(fx_traceback.current_meta) == saved
    assert "ac_graph_id" not in fx_traceback.current_meta


def test_isolated_kernel_trace_keeps_sac_storage_clean():
    storage = _run_checkpointed_eager(isolate=True)
    assert torch.ops.aten.mul.Tensor not in storage
    assert torch.ops.aten.mm.default in storage


def test_unisolated_kernel_trace_breaks_sac_region():
    with pytest.raises(RuntimeError, match="not found in storage"):
        _run_checkpointed_eager(isolate=False)
    assert "ac_graph_id" in fx_traceback.current_meta


def test_kernel_trace_in_sac_region_does_not_crash_partitioner():
    _run_checkpointed_compile(isolate=True)


def test_unisolated_kernel_trace_reproduces_partitioner_crash():
    # A future torch release can guard cleanup_recompute_tags against a None
    # ac_graph_id. The positive test above is the one that must always hold.
    with pytest.raises(Exception, match="NoneType"):
        _run_checkpointed_compile(isolate=False)


class _OpCounter(TorchDispatchMode):
    """Record every op that an ambient mode can see."""

    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return func(*args, **(kwargs or {}))


def test_meta_inference_preserves_other_dispatch_modes():
    kernel = TorchMlirKernel(torch.mul, "aten::mul.Tensor")
    x = torch.randn(4, 4)
    counter = _OpCounter()
    with counter:
        out = kernel.infer_outputs_from_meta((x, x), {})
    assert out.is_meta
    assert torch.ops.aten.mul.Tensor in counter.ops


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("sac_position", [0, 1, 2])
@pytest.mark.parametrize("raise_error", [False, True])
def test_isolated_kernel_trace_preserves_and_restores_other_modes(
    recompute, sac_position, raise_error
):
    sac = create_selective_checkpoint_contexts(_sac_policy)[int(recompute)]
    first, second = _OpCounter(), _OpCounter()
    modes = [first, second]
    modes.insert(sac_position, sac)
    error = (
        pytest.raises(RuntimeError, match="test error") if raise_error else contextlib.nullcontext()
    )

    with contextlib.ExitStack() as stack:
        for mode in modes:
            stack.enter_context(mode)
        saved = object()
        sac.storage[torch.ops.aten.mm.default] = {0: saved}
        sac.func_counter[torch.ops.aten.mm.default] = 1
        with error, _isolated_kernel_trace(), _isolated_kernel_trace():
            assert _get_current_dispatch_mode_stack() == [first, second]
            torch.ops.aten.empty.memory_format([2])
            if raise_error:
                raise RuntimeError("test error")
        assert _get_current_dispatch_mode_stack() == modes
        assert dict(sac.storage) == {torch.ops.aten.mm.default: {0: saved}}
        assert dict(sac.func_counter) == {torch.ops.aten.mm.default: 1}

    assert first.ops == [torch.ops.aten.empty.memory_format]
    assert second.ops == first.ops
