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
    """Run a compiled functional all-reduce check for one rank.

    Args:
        rank (int): Rank whose value initializes the local tensor.
        world_size (int): Number of ranks in the device mesh.
        kwargs (dict): Ignored runner arguments.

    Returns:
        None: This function asserts that the reduced tensor has the expected value.
    """
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    @torch.compile(backend="neuron")
    def fn(x):
        """All-reduce a tensor over the enclosing device mesh.

        Args:
            x (torch.Tensor): Local tensor to reduce.

        Returns:
            torch.Tensor: Functional sum reduction result.
        """
        return funcol.all_reduce(x, reduceOp="sum", group=device_mesh)

    x = torch.ones(10, device="neuron") * rank
    y = fn(x)

    expected = torch.ones(10) * sum(range(world_size))
    torch.testing.assert_close(y.cpu(), expected, rtol=1e-4, atol=1e-4)


def run_distributed_model_test(rank, world_size, kwargs):
    """Run a compiled distributed model check for one rank.

    Args:
        rank (int): Process rank, ignored by this check.
        world_size (int): Number of ranks in the device mesh.
        kwargs (dict): Ignored runner arguments.

    Returns:
        None: This function asserts the compiled output device and shape.

    Raises:
        AssertionError: If the compiled output is not on Neuron or has an unexpected shape.
    """
    device_mesh = DeviceMesh("neuron", list(range(world_size)))

    class DistributedModel(nn.Module):
        """Apply a linear layer followed by distributed all-reduce.

        Args:
            mesh (DeviceMesh): Device mesh used by the functional all-reduce.
        """

        def __init__(self, mesh):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.mesh = mesh

        def forward(self, x):
            """Run the distributed model forward pass.

            Args:
                x (torch.Tensor): Input tensor with a final dimension of size 10.

            Returns:
                torch.Tensor: Sum-reduced linear projection with a final dimension of size 5.
            """
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
    """Run distributed ``torch.compile`` tests with the Dynamo backend."""

    @property
    def distributed_tester(self):
        """Create the two-process distributed test runner.

        Returns:
            DistributedTester: Runner configured with a world size of two.
        """
        return DistributedTester(world_size=2)

    def test_distributed_allreduce(self):
        """Test ``torch.compile`` with functional all-reduce on two processes."""
        self.distributed_tester.run_test(run_distributed_allreduce_test)

    def test_distributed_model(self):
        """Test a compiled model with functional collectives on two processes."""
        self.distributed_tester.run_test(run_distributed_model_test)
