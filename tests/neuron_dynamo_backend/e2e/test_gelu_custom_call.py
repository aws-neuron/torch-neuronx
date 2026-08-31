"""GELU -> AwsNeuronGelu / AwsNeuronGeluApprxTanh / AwsNeuronGeluBackward.

Forward: aten.gelu reaches torch-mlir as torch.aten.gelu and is lowered by
TorchToNeuron (ConvertNeuronGeluOp) to the hardware GELU activation instead of
the erf/tanh polynomial that TorchToStablehlo would otherwise emit:
  approximate="none" -> stablehlo.custom_call @AwsNeuronGelu
  approximate="tanh" -> stablehlo.custom_call @AwsNeuronGeluApprxTanh

Backward: exact (approximate="none") gelu_backward lowers to
@AwsNeuronGeluBackward(self) followed by a multiply with grad_output
(ConvertNeuronGeluBackwardOp), matching NxDI's Gelu.backward. There is no
tanh-backward intrinsic, so approximate="tanh" decomposes instead
(_gelu_backward_fallback in decompositions.py).
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


def _fn(x):
    return F.gelu(x)  # approximate="none" (exact / erf)


def _fn_tanh(x):
    return F.gelu(x, approximate="tanh")


def _gelu_backward_fn(grad_output, x):
    # Direct backward op (approximate="none") so it appears in the graph without
    # autograd/device execution -- lets us assert its lowering compile-only.
    return torch.ops.aten.gelu_backward(grad_output, x)


def _gelu_backward_tanh_fn(grad_output, x):
    return torch.ops.aten.gelu_backward(grad_output, x, approximate="tanh")


def _dumped_stablehlo(debug_dir: str) -> str:
    """Concatenate dumped compile artifacts so we can grep the emitted IR."""
    blobs = []
    for path in glob.glob(os.path.join(debug_dir, "**", "*"), recursive=True):
        if os.path.isfile(path):
            with contextlib.suppress(OSError), open(path, errors="ignore") as fh:
                blobs.append(fh.read())
    return "\n".join(blobs)


@requires_compiler
def test_gelu_emits_custom_call(tmp_path, monkeypatch):
    """Exact GELU (approximate="none") lowers to @AwsNeuronGelu (device-free)."""
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        x = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_fn, backend="neuron", fullgraph=True)(x)

    ir = _dumped_stablehlo(str(tmp_path))
    assert "AwsNeuronGelu" in ir, "exact GELU did not lower to the AwsNeuronGelu custom-call"
    assert "AwsNeuronGeluApprxTanh" not in ir, "exact GELU must not use the tanh target"


@requires_compiler
def test_gelu_tanh_emits_custom_call(tmp_path, monkeypatch):
    """tanh GELU (approximate="tanh") lowers to @AwsNeuronGeluApprxTanh."""
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        x = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_fn_tanh, backend="neuron", fullgraph=True)(x)

    ir = _dumped_stablehlo(str(tmp_path))
    assert (
        "AwsNeuronGeluApprxTanh" in ir
    ), "tanh GELU did not lower to the AwsNeuronGeluApprxTanh custom-call"


@requires_compiler
def test_gelu_backward_emits_custom_call(tmp_path, monkeypatch):
    """Exact gelu_backward lowers to @AwsNeuronGeluBackward (device-free)."""
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        g = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_gelu_backward_fn, backend="neuron", fullgraph=True)(g, x)

    ir = _dumped_stablehlo(str(tmp_path))
    assert (
        "AwsNeuronGeluBackward" in ir
    ), "exact gelu_backward did not lower to the AwsNeuronGeluBackward custom-call"


@requires_compiler
def test_gelu_tanh_backward_decomposes(tmp_path, monkeypatch):
    """tanh gelu_backward decomposes -- no custom call (no tanh-backward intrinsic)."""
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        g = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_gelu_backward_tanh_fn, backend="neuron", fullgraph=True)(g, x)

    ir = _dumped_stablehlo(str(tmp_path))
    assert (
        "AwsNeuronGeluBackward" not in ir
    ), "tanh gelu_backward must decompose, not lower to AwsNeuronGeluBackward"


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
    ],
)
def test_gelu_forward_correctness(dtype, rtol, atol):
    """Exact-GELU custom-call output matches the CPU reference."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 128, HIDDEN, dtype=dtype, generator=gen)

    ref = _fn(x)
    with torch.no_grad():
        out = torch.compile(_fn, backend="neuron", fullgraph=True)(x.to("neuron"))
    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_gelu_tanh_forward_correctness(dtype, rtol, atol):
    """tanh-GELU custom-call output matches the CPU reference."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 128, HIDDEN, dtype=dtype, generator=gen)

    ref = _fn_tanh(x)
    with torch.no_grad():
        out = torch.compile(_fn_tanh, backend="neuron", fullgraph=True)(x.to("neuron"))
    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_gelu_backward_correctness(dtype, rtol, atol):
    """Gradients flow correctly through the GELU decomposition backward.

    Forward is the @AwsNeuronGelu custom-call; the backward decomposes (for now).
    Checks the compiled input grad matches the CPU autograd reference -- i.e. the
    forward custom-call composes with the decomposed backward end to end.
    """
    gen = torch.Generator().manual_seed(0)
    xc = torch.randn(4, HIDDEN, dtype=dtype, generator=gen).requires_grad_()
    x = xc.detach().to("neuron").requires_grad_()

    torch.compile(_fn, backend="neuron", fullgraph=True)(x).pow(2).sum().backward()
    _fn(xc).pow(2).sum().backward()

    assert x.grad is not None, "no gradient propagated to the input"
    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_module_matches_cpu(approximate):
    """nn.GELU module path compiles and matches CPU for forward AND backward."""
    gen = torch.Generator().manual_seed(0)
    m = torch.nn.GELU(approximate=approximate)
    xc = torch.randn(4, 128, HIDDEN, dtype=torch.float32, generator=gen).requires_grad_()
    x = xc.detach().to("neuron").requires_grad_()

    out = torch.compile(m.to("neuron"), backend="neuron", fullgraph=True)(x)
    ref = m(xc)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)

    out.pow(2).sum().backward()
    ref.pow(2).sum().backward()
    assert x.grad is not None, "no gradient propagated through nn.GELU"
    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=1e-3, atol=1e-3)
