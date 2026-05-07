import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from torch_neuronx.neuron_dynamo_backend import set_model_name


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset dynamo compilation cache before each test"""
    torch._dynamo.reset()


# Define functional collective operations to test
COLLECTIVE_OPERATIONS = [
    ("all_reduce_sum", lambda x, mesh: funcol.all_reduce(x, reduceOp="sum", group=mesh)),
    ("all_reduce_avg", lambda x, mesh: funcol.all_reduce(x, reduceOp="avg", group=mesh)),
    ("all_reduce_min", lambda x, mesh: funcol.all_reduce(x, reduceOp="min", group=mesh)),
    ("all_reduce_max", lambda x, mesh: funcol.all_reduce(x, reduceOp="max", group=mesh)),
    ("all_gather_tensor", lambda x, mesh: funcol.all_gather_tensor(x, gather_dim=0, group=mesh)),
    (
        "reduce_scatter_tensor",
        lambda x, mesh: funcol.reduce_scatter_tensor(x, "sum", scatter_dim=0, group=mesh),
    ),
    (
        "all_reduce_coalesced",
        lambda x, mesh: funcol.all_reduce_coalesced([x], reduceOp="sum", group=mesh)[0],
    ),
    (
        "all_gather_into_tensor_coalesced",
        lambda x, mesh: funcol.all_gather_into_tensor_coalesced([x], group=mesh)[0],
    ),
    (
        "reduce_scatter_tensor_coalesced",
        lambda x, mesh: funcol.reduce_scatter_tensor_coalesced(
            [x], "sum", scatter_dim=[0], group=mesh
        )[0],
    ),
    # (
    #     "all_to_all",
    #     lambda x, mesh: funcol.all_to_all_single(
    #         x, output_split_sizes=None, input_split_sizes=None, group=mesh
    #     ),
    # ),
]


def _test_single_collective(
    collective_name, collective_fn, rank, world_size, process_group, device
):
    """Test a single collective operation"""
    set_model_name(f"{collective_name}_rank{rank}")

    x = torch.tensor([rank + 1.0, rank + 2.0, rank + 3.0, rank + 4.0] * world_size, device=device)

    eager_result = collective_fn(x.clone(), process_group)
    compiled_fn = torch.compile(lambda t: collective_fn(t, process_group), backend="neuron")
    compiled_result = compiled_fn(x.clone())

    torch.neuron.synchronize()

    torch.testing.assert_close(eager_result.cpu(), compiled_result.cpu(), atol=1e-5, rtol=1e-4)


def run_permute_tensor_test(rank, world_size, kwargs):
    """Test permute_tensor collective operation with torch.compile"""
    device = torch.neuron.current_device()
    process_group = dist.group.WORLD

    set_model_name(f"permute_tensor_rank{rank}")

    x = torch.tensor([rank + 1.0, rank + 2.0, rank + 3.0, rank + 4.0] * world_size, device=device)
    permutation = list(range(world_size))
    permutation.reverse()

    cpu_pg = dist.new_group(backend="gloo")
    eager_result = funcol.permute_tensor(x.cpu(), permutation, group=cpu_pg)
    compiled_fn = torch.compile(
        lambda t: funcol.permute_tensor(t, permutation, group=process_group), backend="neuron"
    )
    compiled_result = compiled_fn(x.clone())

    torch.neuron.synchronize()

    torch.testing.assert_close(eager_result, compiled_result.cpu(), atol=1e-5, rtol=0)


def run_functional_collectives_test(rank, world_size, kwargs):
    """Test all functional collective operations with torch.compile"""
    device = torch.device(f"neuron:{rank}")
    process_group = dist.group.WORLD

    for collective_name, collective_fn in COLLECTIVE_OPERATIONS:
        _test_single_collective(
            collective_name, collective_fn, rank, world_size, process_group, device
        )
        dist.barrier(device_ids=[rank])


class TestFunctionalCollectives(BaseCollectiveOpTest):
    """Test class for functional collectives unit tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_compile_functional_collectives(self, monkeypatch):
        """Test functional collective operations with torch.compile."""
        monkeypatch.setenv("TORCH_NEURONX_MLIR_ATEN_OPS", "1")
        self.distributed_tester.run_test(run_functional_collectives_test)

    @pytest.mark.multi_device
    def test_compile_permute_tensor(self, monkeypatch):
        """Test permute_tensor collective operation with torch.compile."""
        monkeypatch.setenv("TORCH_NEURONX_MLIR_ATEN_OPS", "1")
        self.distributed_tester.run_test(run_permute_tensor_test)
