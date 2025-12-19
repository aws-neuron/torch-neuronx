"""
Distributed tests for torch.compile with neuron backend

Run with: pytest examples/torch_compile/test_distributed.py
"""

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from .utils.distributed import DistributedTester


def run_distributed_allreduce_test(rank, world_size, kwargs):
    """Test torch.compile with functional all-reduce"""
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    @torch.compile(backend="neuron")
    def fn(x):
        return funcol.all_reduce(x, reduceOp="sum", group=device_mesh)

    x = torch.ones(10, device="neuron") * rank
    y = fn(x)

    expected = torch.ones(10) * sum(range(world_size))
    torch.testing.assert_close(y.cpu(), expected, rtol=1e-4, atol=1e-4)


def run_distributed_model_test(rank, world_size, kwargs):
    """Test compiled model with functional collectives"""
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    class DistributedModel(nn.Module):
        def __init__(self, mesh):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.mesh = mesh

        def forward(self, x):
            x = self.linear(x)
            x = funcol.all_reduce(x, reduceOp="sum", group=self.mesh)
            return x

    model = DistributedModel(device_mesh).to("neuron")
    compiled_model = torch.compile(model, backend="neuron")

    x = torch.randn(2, 10, device="neuron")
    y = compiled_model(x)

    assert y.device.type == "neuron"
    assert y.shape == (2, 5)


class TestTorchCompileDistributed:
    """Test trivial cases for torch.compile with neuron backend in distributed setting."""

    @property
    def distributed_tester(self):
        return DistributedTester(world_size=2)

    def test_distributed_allreduce(self):
        """Test torch.compile with functional all-reduce on 2 processes"""
        self.distributed_tester.run_test(run_distributed_allreduce_test)

    def test_distributed_model(self):
        """Test compiled model with functional collectives on 2 processes"""
        self.distributed_tester.run_test(run_distributed_model_test)
