"""RMSNorm -> AwsNeuronRmsNorm custom-call lowering.

Forward F.rms_norm / aten._fused_rms_norm is kept out of the decomposition table
so it survives to torch-mlir as torch.operator "torch.aten._fused_rms_norm" and is
lowered to stablehlo.custom_call @AwsNeuronRmsNorm in TorchToNeuron. Backward stays
in the decomposition table (decomposes); the forward's TorchToNeuron lowering
recomputes rstd so the decomposed backward gets a correct value -> correct grads.
"""

import contextlib
import glob
import os

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from tests.neuron_dynamo_backend.compile_only_utils import (
    COMPILE_ONLY_TARGET,
    compile_only_platform_env,
    requires_compiler,
)

HIDDEN = 3072


def _fn(x, w):
    return F.rms_norm(x, (x.shape[-1],), w, 1e-6)


def _fn_multidim(x, w):
    # normalized_shape spans the last TWO dims -> not expressible by the
    # single-last-dim AwsNeuronRmsNorm kernel, so this must take the
    # decomposition fallback rather than the custom-call.
    return F.rms_norm(x, (x.shape[-2], x.shape[-1]), w, 1e-6)


def _dumped_stablehlo(debug_dir: str) -> str:
    """Concatenate dumped compile artifacts so we can grep the emitted IR."""
    blobs = []
    for path in glob.glob(os.path.join(debug_dir, "**", "*"), recursive=True):
        if os.path.isfile(path):
            with contextlib.suppress(OSError), open(path, errors="ignore") as fh:
                blobs.append(fh.read())
    return "\n".join(blobs)


@requires_compiler
def test_rms_norm_emits_custom_call(tmp_path, monkeypatch):
    """Forward RMSNorm is lowered to the AwsNeuronRmsNorm custom-call (device-free).

    Inference (no_grad): rstd is dead, so the graph reduces to the single
    custom-call.
    """
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        w = torch.ones(HIDDEN, dtype=torch.bfloat16, device="neuron")
        x = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_fn, backend="neuron", fullgraph=True)(x, w)

    ir = _dumped_stablehlo(str(tmp_path))
    assert "AwsNeuronRmsNorm" in ir, "forward did not lower to the AwsNeuronRmsNorm custom-call"


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
    ],
)
def test_rms_norm_forward_correctness(dtype, rtol, atol):
    """Custom-call output matches the CPU reference."""
    torch.manual_seed(0)
    x = torch.randn(4, 128, HIDDEN, dtype=dtype)
    w = torch.randn(HIDDEN, dtype=dtype)

    ref = _fn(x, w)
    with torch.no_grad():
        out = torch.compile(_fn, backend="neuron", fullgraph=True)(x.to("neuron"), w.to("neuron"))
    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)


@requires_compiler
def test_rms_norm_multidim_decomposes(tmp_path, monkeypatch):
    """Multi-dim normalized_shape takes the decomposition fallback, not the kernel.

    AwsNeuronRmsNorm normalizes exactly the last dim, so a multi-trailing-dim
    RMSNorm is decomposed (in decompositions.py) instead of lowering to the
    custom-call.
    """
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        w = torch.ones(8, 16, dtype=torch.bfloat16, device="neuron")
        x = torch.randn(4, 8, 16, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_fn_multidim, backend="neuron", fullgraph=True)(x, w)

    ir = _dumped_stablehlo(str(tmp_path))
    assert (
        "AwsNeuronRmsNorm" not in ir
    ), "multi-dim RMSNorm must decompose, not lower to the last-dim-only kernel"


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
def test_rms_norm_multidim_forward_correctness():
    """Multi-dim decomposition output matches the CPU reference."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)

    ref = _fn_multidim(x, w)
    with torch.no_grad():
        out = torch.compile(_fn_multidim, backend="neuron", fullgraph=True)(
            x.to("neuron"), w.to("neuron")
        )
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
def test_rms_norm_multidim_backward_correctness():
    """Multi-dim grads match CPU: AOTAutograd differentiates the decomposed
    primitives directly (no fused backward / rstd dependency)."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, dtype=torch.float32, device="neuron", requires_grad=True)
    w = torch.randn(8, 16, dtype=torch.float32, device="neuron", requires_grad=True)
    xc = x.detach().cpu().requires_grad_()
    wc = w.detach().cpu().requires_grad_()

    torch.compile(_fn_multidim, backend="neuron", fullgraph=True)(x, w).pow(2).sum().backward()
    _fn_multidim(xc, wc).pow(2).sum().backward()

    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w.grad.cpu(), wc.grad, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
def test_rms_norm_backward_correctness():
    """Grads match the CPU reference: the forward's recomputed rstd feeds the
    decomposed backward correctly (fp32 for a tight gradient check)."""
    torch.manual_seed(0)
    x = torch.randn(4, HIDDEN, dtype=torch.float32, device="neuron", requires_grad=True)
    w = torch.randn(HIDDEN, dtype=torch.float32, device="neuron", requires_grad=True)
    xc = x.detach().cpu().requires_grad_()
    wc = w.detach().cpu().requires_grad_()

    torch.compile(_fn, backend="neuron", fullgraph=True)(x, w).pow(2).sum().backward()
    _fn(xc, wc).pow(2).sum().backward()

    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w.grad.cpu(), wc.grad, rtol=1e-3, atol=1e-3)


@requires_compiler
def test_rms_norm_affine_free_decomposes(tmp_path, monkeypatch):
    """weight=None (elementwise_affine=False) can't use the kernel, so it must
    decompose -- compile cleanly and NOT emit the custom-call.

    Regression guard: the kernel requires a real weight, so this path (like
    eps=None and multi-dim) is handled by the decomposition fallback rather than
    hard-failing legalization on the surviving torch.operator.
    """
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    m = torch.nn.RMSNorm(HIDDEN, elementwise_affine=False).to("neuron")
    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(m, backend="neuron", fullgraph=True)(x)

    ir = _dumped_stablehlo(str(tmp_path))
    assert (
        "AwsNeuronRmsNorm" not in ir
    ), "affine-free RMSNorm (weight=None) must decompose, not hit the kernel"


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_rms_norm_module_matches_cpu(elementwise_affine):
    """nn.RMSNorm module path compiles and matches CPU for both affine and
    affine-free. The module's default eps=None and (affine-free) weight=None are
    the previously-regressed cases now covered by the decomposition fallback.
    """
    torch.manual_seed(0)
    m = torch.nn.RMSNorm(HIDDEN, elementwise_affine=elementwise_affine, dtype=torch.float32)
    if elementwise_affine:
        m.weight.data.normal_()
    x = torch.randn(4, 128, HIDDEN, dtype=torch.float32)

    ref = m(x)
    with torch.no_grad():
        out = torch.compile(m.to("neuron"), backend="neuron", fullgraph=True)(x.to("neuron"))
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)
