"""Tests for PrivateUse1 functional collective dispatches.

Verifies that _c10d_functional::all_gather_into_tensor,
_c10d_functional::reduce_scatter_tensor and _c10d_functional::all_reduce
route through the Neuron PrivateUse1 implementations registered in
functional_collectives.py.

Run with: torchrun --nproc_per_node=2 -m pytest \
    tests/distributed/collective_ops/test_functional_collective_dispatch.py -v
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

import torch_neuronx

pytestmark = pytest.mark.skipif(
    not os.environ.get("WORLD_SIZE") or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="Requires distributed environment with WORLD_SIZE >= 2",
)


@pytest.fixture(scope="module")
def dist_setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="neuron", timeout=timedelta(seconds=30))
    yield


class TestFunctionalCollectiveDispatch:
    """Test PrivateUse1 dispatches for functional collectives."""

    def test_all_gather_tensor(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.full((4,), float(rank), device="neuron")
        result = funcol.all_gather_tensor(input_tensor, gather_dim=0, group=dist.group.WORLD)

        expected = torch.cat([torch.full((4,), float(r)) for r in range(size)])
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_gather_tensor_2d(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.full((2, 3), float(rank), device="neuron")
        result = funcol.all_gather_tensor(input_tensor, gather_dim=0, group=dist.group.WORLD)

        expected = torch.cat([torch.full((2, 3), float(r)) for r in range(size)])
        torch.testing.assert_close(result.cpu(), expected)

    def test_reduce_scatter_tensor_sum(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.ones(size * 4, device="neuron") * (rank + 1)
        result = funcol.reduce_scatter_tensor(
            input_tensor, "sum", scatter_dim=0, group=dist.group.WORLD
        )

        expected_val = float(sum(range(1, size + 1)))
        expected = torch.full((4,), expected_val)
        torch.testing.assert_close(result.cpu(), expected)

    def test_reduce_scatter_tensor_avg(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.ones(size * 4, device="neuron") * (rank + 1)
        result = funcol.reduce_scatter_tensor(
            input_tensor, "avg", scatter_dim=0, group=dist.group.WORLD
        )

        expected_val = float(sum(range(1, size + 1))) / size
        expected = torch.full((4,), expected_val)
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_reduce_sum(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.full((4,), float(rank + 1), device="neuron")
        result = funcol.all_reduce(input_tensor, "sum", group=dist.group.WORLD)

        expected_val = float(sum(range(1, size + 1)))
        expected = torch.full((4,), expected_val)
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_reduce_avg(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.full((4,), float(rank + 1), device="neuron")
        result = funcol.all_reduce(input_tensor, "avg", group=dist.group.WORLD)

        expected_val = float(sum(range(1, size + 1))) / size
        expected = torch.full((4,), expected_val)
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_reduce_min(self, dist_setup):
        rank = dist.get_rank()

        input_tensor = torch.full((4,), float(rank + 1), device="neuron")
        result = funcol.all_reduce(input_tensor, "min", group=dist.group.WORLD)

        expected = torch.full((4,), 1.0)
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_reduce_max(self, dist_setup):
        rank = dist.get_rank()
        size = dist.get_world_size()

        input_tensor = torch.full((4,), float(rank + 1), device="neuron")
        result = funcol.all_reduce(input_tensor, "max", group=dist.group.WORLD)

        expected = torch.full((4,), float(size))
        torch.testing.assert_close(result.cpu(), expected)

    def test_all_reduce_coalesced_sum(self, dist_setup):
        """Functional coalesced variant: inputs unmodified, reduced clones returned."""
        rank = dist.get_rank()
        size = dist.get_world_size()

        inputs = [
            torch.full((4,), float(rank + 1), device="neuron"),
            torch.full((2, 3), float(rank + 1) * 10, device="neuron"),
        ]
        results = funcol.all_reduce_coalesced(inputs, "sum", group=dist.group.WORLD)

        expected_val = float(sum(range(1, size + 1)))
        torch.testing.assert_close(results[0].cpu(), torch.full((4,), expected_val))
        torch.testing.assert_close(results[1].cpu(), torch.full((2, 3), expected_val * 10))
        torch.testing.assert_close(inputs[0].cpu(), torch.full((4,), float(rank + 1)))
