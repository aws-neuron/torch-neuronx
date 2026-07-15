"""End-to-end broadcast under torch.compile(backend="neuron").

Exercises the full pipeline (dynamo -> AOT -> CollectiveLegalization ->
StableHLO -> NEFF) for a broadcast on a strict subgroup, comparing the compiled
result against eager.

Run with enough visible cores for world_size=4:

    NEURON_RT_VISIBLE_CORES=0-3 ./tools/run-test \
        tests/distributed/neuron_dynamo_backend/collectives/test_broadcast_compile.py -v
"""

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import init_device_mesh

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from torch_neuronx.neuron_dynamo_backend import set_model_name


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset dynamo compilation cache before each test"""
    torch._dynamo.reset()


def run_broadcast_subgroup_test(rank, world_size, kwargs):
    """Broadcast on a 2x2 mesh tp subgroup; compiled must match eager."""
    device = torch.device(f"neuron:{rank}")
    mesh = init_device_mesh("neuron", (2, 2), mesh_dim_names=("dp", "tp"))
    tp_group = (mesh, 1)  # inner dim -> groups [[0, 1], [2, 3]]

    set_model_name(f"broadcast_subgroup_rank{rank}")

    x = torch.full((8,), float(rank + 1), device=device)

    eager_result = funcol.broadcast(x.clone(), src=0, group=tp_group)
    compiled_fn = torch.compile(
        lambda t: funcol.broadcast(t, src=0, group=tp_group), backend="neuron"
    )
    compiled_result = compiled_fn(x.clone())

    torch.neuron.synchronize()

    torch.testing.assert_close(eager_result.cpu(), compiled_result.cpu(), atol=1e-5, rtol=1e-4)
    dist.barrier(device_ids=[rank])


class TestBroadcastCompile(BaseCollectiveOpTest):
    """torch.compile subgroup broadcast test using DistributedTester."""

    @property
    def world_size(self):
        return 4

    @pytest.mark.multi_device
    def test_compile_broadcast_subgroup(self):
        """Broadcast on a tp subgroup lowers and matches eager under compile."""
        self.distributed_tester.run_test(run_broadcast_subgroup_test)
