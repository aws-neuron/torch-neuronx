"""torch.compile end-to-end for aten.mish and its backward.

aten.mish decomposes (torch-mlir DecomposeAtenMishOp) into tanh + softplus on
the compile path. aten.mish_backward is added to the neuron decomposition table
(decompositions.py) so it also decomposes to primitives instead of reaching
StableHLO as an opaque operator. These tests check that both the forward and the
autograd backward compile with fullgraph (no graph break) and match CPU.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

HIDDEN = 3072


def _fn(x):
    return F.mish(x)


@pytest.mark.skipif(not torch.neuron.is_available(), reason="Neuron device required")
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float16, 1e-2, 1e-2),
    ],
)
def test_mish_forward_correctness(dtype, rtol, atol):
    """Compiled forward output matches the CPU reference."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(4, 128, HIDDEN, dtype=dtype, generator=gen)

    ref = _fn(x)
    torch._dynamo.reset()
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
def test_mish_backward_correctness(dtype, rtol, atol):
    """The compiled autograd backward matches the CPU reference.

    The backward reaches aten.mish_backward, which the neuron decomposition
    table lowers to primitives. fullgraph=True fails if the backward graph
    breaks.
    """
    gen = torch.Generator().manual_seed(0)
    xc = torch.randn(4, HIDDEN, dtype=dtype, generator=gen).requires_grad_()
    x = xc.detach().to("neuron").requires_grad_()

    torch._dynamo.reset()
    torch.compile(_fn, backend="neuron", fullgraph=True)(x).pow(2).sum().backward()
    _fn(xc).pow(2).sum().backward()

    assert x.grad is not None, "no gradient propagated to the input"
    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=rtol, atol=atol)
