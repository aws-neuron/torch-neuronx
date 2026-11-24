"""Regression for softmax reductions whose last 512-wide chunk is partial.

neuronx-cc's custom softmax (native-to-custom-softmax) returned wrong rows for reduction sizes
between 512 and 2048 that are not a multiple of 512 (fp32: 513..1023; bf16 and fp16 also above
1024 and 1536), forward and backward, eager and compiled. Every other row was wrong at 1024 rows.
"""

import pytest
import torch

import torch_neuronx

TOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2, torch.float16: 2e-3}


def _fwd_bwd(x, w):
    x = x.detach().requires_grad_(True)
    y = torch.softmax(x, -1)
    (y.float() * w).sum().backward()
    return y.detach(), x.grad


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compile"])
@pytest.mark.parametrize("dtype", list(TOL), ids=["fp32", "bf16", "fp16"])
@pytest.mark.parametrize("size", [512, 640, 1023, 1024, 1025, 1300, 1800, 2048])
def test_softmax_partial_last_chunk(size, dtype, compiled):
    g = torch.Generator().manual_seed(size)
    x = torch.randn(1024, size, generator=g).to(dtype)
    w = torch.randn(1024, size, generator=g)
    fn = torch.compile(_fwd_bwd, backend="neuron", dynamic=False) if compiled else _fwd_bwd

    y, grad = fn(x.to("neuron"), w.to("neuron"))
    y_ref, grad_ref = _fwd_bwd(x.float(), w)

    tol = TOL[dtype]
    torch.testing.assert_close(y.float().cpu(), y_ref, atol=tol, rtol=0)
    torch.testing.assert_close(
        grad.float().cpu(), grad_ref, atol=tol * max(1.0, grad_ref.abs().max().item()), rtol=0
    )
