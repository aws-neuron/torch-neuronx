"""Basic dynamic shape tests"""

import pytest
import torch


@pytest.mark.xfail(reason="stablehlo.broadcast_in_dim does not support dynamic shapes")
def test_add_dynamic_batch():
    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    model = AddOne()
    compiled = torch.compile(model, backend="neuron")

    # First call with batch=2, marking dim 0 as dynamic
    x1 = torch.randn(2, 4)
    torch._dynamo.mark_dynamic(x1, 0)
    cpu_out1 = model(x1)
    interp_out1 = compiled(x1)
    torch.testing.assert_close(interp_out1, cpu_out1, rtol=1e-5, atol=1e-5)

    # Second call with different batch size — no recompilation needed
    x2 = torch.randn(5, 4)
    torch._dynamo.mark_dynamic(x2, 0)
    cpu_out2 = model(x2)
    interp_out2 = compiled(x2)
    torch.testing.assert_close(interp_out2, cpu_out2, rtol=1e-5, atol=1e-5)
