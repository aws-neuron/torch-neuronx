"""Check SAC storage and gradients across cold and warm kernel caches."""

import pytest
import torch
import torch.fx.traceback as fx_traceback
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

import torch_neuronx
from torch_neuronx.python_ops.compilation import hlo_cache
from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)


@pytest.fixture(autouse=True)
def _isolate_trace_state(monkeypatch):
    saved_meta = dict(fx_traceback.current_meta)
    monkeypatch.setattr(hlo_cache, "get", lambda key: None)
    monkeypatch.setattr(hlo_cache, "cache_enabled", lambda **kwargs: False)
    try:
        yield
    finally:
        fx_traceback.current_meta = saved_meta


def _sac_policy(ctx, func, *args, **kwargs):
    if func is torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def test_hlo_generation_leaves_sac_state_unchanged():
    kernel = TorchMlirKernel(torch.mm, "aten::mm")
    x = torch.ones(8, 8, device="neuron")
    forward, _ = create_selective_checkpoint_contexts(_sac_policy)
    saved_meta = dict(fx_traceback.current_meta)

    with forward:
        result = kernel._generate_hlo((x, x), {})

    assert result.hlo_bytes
    assert not forward.storage
    assert not forward.func_counter
    assert fx_traceback.current_meta == saved_meta


@pytest.mark.parametrize("warm_cache", [False, True], ids=["cold", "warm"])
def test_sac_kernel_cache_preserves_outputs_and_gradients(monkeypatch, warm_cache):
    kernel = TorchMlirKernel(torch.mm, "aten::mm")
    generate_hlo = kernel._generate_hlo
    compilations = []

    def count_compilation(*args, **kwargs):
        compilations.append(1)
        return generate_hlo(*args, **kwargs)

    monkeypatch.setattr(kernel, "_generate_hlo", count_compilation)

    def compute(a, b):
        # Equal shapes and different values expose incorrect SAC cache entries.
        return (a @ b).sin() + (a @ (b * 2)).cos()

    def block(a, b):
        kernel._infer_output_dtypes_impl(a, b)
        kernel._get_or_compile_hlo((a, b), {})
        return compute(a, b)

    x_cpu = torch.linspace(-0.4, 0.6, 64).reshape(8, 8).requires_grad_()
    w_cpu = torch.linspace(-0.3, 0.2, 64).reshape(8, 8).requires_grad_()
    expected = compute(x_cpu, w_cpu)
    expected.sum().backward()
    x = x_cpu.detach().to("neuron").requires_grad_()
    w = w_cpu.detach().to("neuron").requires_grad_()

    if warm_cache:
        kernel._infer_output_dtypes_impl(x, w)
        kernel._get_or_compile_hlo((x, w), {})
    assert len(compilations) == int(warm_cache)

    actual = checkpoint(
        block,
        x,
        w,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy),
    )
    actual.sum().backward()

    assert len(compilations) == 1
    assert x.grad is not None
    assert w.grad is not None
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(w.grad.cpu(), w_cpu.grad, rtol=1e-4, atol=1e-5)
