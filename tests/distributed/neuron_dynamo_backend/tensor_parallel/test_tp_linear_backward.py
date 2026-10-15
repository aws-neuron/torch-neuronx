"""
Regression test for DTensor + torch.compile linear_backward sharding propagation.

Tests that linear_backward sharding works correctly with both bias=True and bias=False.
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest


def run_linear_backward_test(rank, world_size, kwargs):
    """Test DTensor + torch.compile backward for linear layer with CPU accuracy check."""
    device = f"neuron:{rank}"
    bias = kwargs["bias"]
    mesh = init_device_mesh("neuron", (world_size,), mesh_dim_names=("tp",))

    # Create CPU reference model
    torch.manual_seed(42)
    cpu_model = nn.Linear(64, 128, bias=bias)
    x_cpu = torch.randn(2, 64, requires_grad=True)

    # CPU forward/backward
    y_cpu = cpu_model(x_cpu)
    y_cpu.sum().backward()
    ref_grad = x_cpu.grad.clone()

    # Neuron model with same weights
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 128, bias=bias, device=device)

        def forward(self, x):
            return self.linear(x)

    torch.manual_seed(42)
    model = Model()
    parallelize_module(model, mesh["tp"], {"linear": ColwiseParallel()})

    x = x_cpu.detach().clone().to(device).requires_grad_(True)
    compiled = torch.compile(model, backend="neuron", fullgraph=True)
    y = compiled(x)
    y.sum().backward()

    # Shape checks
    assert y.shape == (2, 128 // world_size)
    assert x.grad is not None
    assert x.grad.shape == (2, 64)

    # Accuracy check against CPU reference
    torch.testing.assert_close(x.grad.cpu(), ref_grad, rtol=1e-4, atol=1e-4)


class TestTPLinearBackward(BaseCollectiveOpTest):
    """Regression tests for linear_backward sharding with tensor parallelism."""

    @pytest.mark.multi_device
    @pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "with_bias"])
    def test_linear_backward(self, bias):
        """Test linear_backward sharding propagation."""
        self.distributed_tester.run_test(run_linear_backward_test, bias=bias)
