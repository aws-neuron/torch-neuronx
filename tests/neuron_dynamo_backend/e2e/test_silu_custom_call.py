"""SiLU -> AwsNeuronSilu custom-call lowering (forward only).

aten.silu is kept OUT of the decomposition table (decompositions.py) so it
survives to torch-mlir as torch.aten.silu and lowers to
stablehlo.custom_call @AwsNeuronSilu (TorchToNeuron, ConvertNeuronSiluOp). The
compiler maps it to the __builtin_silu hardware activation.
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
    return F.silu(x)


def _silu_backward_fn(grad_output, x):
    # Call the backward op directly so it appears in the graph without needing
    # autograd/device execution -- lets us assert its lowering compile-only.
    return torch.ops.aten.silu_backward(grad_output, x)


def _dumped_stablehlo(debug_dir: str) -> str:
    """Concatenate dumped compile artifacts so we can grep the emitted IR."""
    blobs = []
    for path in glob.glob(os.path.join(debug_dir, "**", "*"), recursive=True):
        if os.path.isfile(path):
            with contextlib.suppress(OSError), open(path, errors="ignore") as fh:
                blobs.append(fh.read())
    return "\n".join(blobs)


@requires_compiler
def test_silu_emits_custom_call(tmp_path, monkeypatch):
    """Forward SiLU is lowered to the AwsNeuronSilu custom-call (device-free)."""
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
    assert "AwsNeuronSilu" in ir, "forward did not lower to the AwsNeuronSilu custom-call"


@requires_compiler
def test_silu_backward_decomposes(tmp_path, monkeypatch):
    """aten.silu_backward is decomposed, NOT lowered to a custom-call.

    Regression guard: @AwsNeuronSiluBackward does not compile on neuronx-cc, and
    NxDI does not emit it, so silu_backward must take the decomposition path.
    """
    for k, v in compile_only_platform_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TORCH_NEURONX_DEBUG_DIR", str(tmp_path))

    from torch_neuronx.compiler import CompileOptions, compile_only

    torch._dynamo.reset()
    with compile_only(CompileOptions(target=COMPILE_ONLY_TARGET)):
        g = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="neuron")
        with torch.no_grad():
            torch.compile(_silu_backward_fn, backend="neuron", fullgraph=True)(g, x)

    ir = _dumped_stablehlo(str(tmp_path))
    assert (
        "AwsNeuronSiluBackward" not in ir
    ), "silu_backward must decompose, not lower to the (non-compiling) custom-call"


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
    ],
)
def test_silu_forward_correctness(dtype, rtol, atol):
    """Custom-call forward output matches the CPU reference."""
    # Seed a CPU generator (not torch.manual_seed, which would also seed the
    # Neuron device RNG and trigger the on-device seed kernel).
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
def test_silu_backward_correctness(dtype, rtol, atol):
    """Gradients flow correctly through the SiLU decomposition backward.

    The forward is the @AwsNeuronSilu custom-call; the backward decomposes. This
    checks the compiled input grad matches the CPU autograd reference, i.e. the
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
def test_silu_module_matches_cpu():
    """nn.SiLU module path compiles and matches CPU for forward AND backward."""
    gen = torch.Generator().manual_seed(0)
    m = torch.nn.SiLU()
    xc = torch.randn(4, 128, HIDDEN, dtype=torch.float32, generator=gen).requires_grad_()
    x = xc.detach().to("neuron").requires_grad_()

    out = torch.compile(m.to("neuron"), backend="neuron", fullgraph=True)(x)
    ref = m(xc)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    out.pow(2).sum().backward()
    ref.pow(2).sum().backward()
    assert x.grad is not None, "no gradient propagated through nn.SiLU"
    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=1e-3, atol=1e-3)
