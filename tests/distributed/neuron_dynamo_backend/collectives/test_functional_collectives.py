"""
Test functional collective operations with torch.compile

This script tests the functional collective operations with the Neuron backend.

Usage:
    # Set artifacts directory (optional, defaults to neuron_backend_artifacts/)
    export TORCH_NEURONX_DEBUG_DIR=./test_artifacts/functional_collectives

    # Run with pytest
    pytest tests/neuron_dynamo_backend/component/collectives/test_functional_collectives.py

* trn1 takes 2, 8, 16, 32 as --nproc_per_node
"""

import logging
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest

# Import the neuron backend
from tests.distributed.utils import DistributedTester
from torch_neuronx.neuron_dynamo_backend import set_model_name

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disable verbose logging for cleaner output
torch._logging.set_logs(
    # aot_graphs=True,
    # graph_code=True,
    # graph_breaks=True
)

torch._dynamo.config.automatic_dynamic_shapes = False


# Define collective operations to test with their configurations
# Each entry: (name, callable, kwargs, needs_world_size_tensor)
COLLECTIVE_OPERATIONS = [
    # Reduction operations with different reduce ops
    # Note: PyTorch functional collectives use "sum", "avg", "product", "min", "max"
    ("all_reduce_sum", lambda x, mesh: funcol.all_reduce(x, reduceOp="sum", group=mesh), {}, False),
    ("all_reduce_avg", lambda x, mesh: funcol.all_reduce(x, reduceOp="avg", group=mesh), {}, False),
    # ("all_reduce_product", lambda x, mesh: funcol.all_reduce(x, reduceOp="product", group=mesh), \
    #   {}, False),  # NOTE: not supported by neuronx-cc
    ("all_reduce_min", lambda x, mesh: funcol.all_reduce(x, reduceOp="min", group=mesh), {}, False),
    ("all_reduce_max", lambda x, mesh: funcol.all_reduce(x, reduceOp="max", group=mesh), {}, False),
    (
        "all_gather_tensor",
        lambda x, mesh: funcol.all_gather_tensor(x, gather_dim=0, group=mesh),
        {},
        False,
    ),
    # TODO: Not implemented yet
    # ("broadcast", lambda x, mesh: funcol.broadcast(x, src=0, group=mesh), {}, False),
    (
        "reduce_scatter_tensor",
        lambda x, mesh: funcol.reduce_scatter_tensor(x, "sum", scatter_dim=0, group=(mesh, 0)),
        {},
        True,
    ),
    # Coalesced operations (take lists)
    # ("all_reduce_coalesced", lambda x, \
    #   mesh: funcol.all_reduce_coalesced([x], reduceOp="sum", group=mesh)[0], {}, False),
    # ("all_gather_into_tensor_coalesced", lambda x, \
    #   mesh: funcol.all_gather_into_tensor_coalesced([x], group=mesh)[0], {}, False),
    # ("reduce_scatter_tensor_coalesced", lambda x, \
    #   mesh: funcol.reduce_scatter_tensor_coalesced([x], "sum", scatter_dim=[0], group=mesh)[0], \
    #   {}, True),
]


def _run_single_collective(collective_config, rank, world_size, device_mesh, device):
    """Test a single collective operation"""
    func_name, collective_fn, kwargs, needs_world_size_tensor = collective_config

    logger.debug(f"Rank {rank}: Testing {func_name}...")

    # Set model name for artifact tracking
    set_model_name(f"{func_name}_rank{rank}")

    # Create appropriate test tensor based on the operation
    if needs_world_size_tensor:
        # These need tensors divisible by world_size
        x = torch.tensor(
            [rank + 1.0, rank + 2.0, rank + 3.0, rank + 4.0] * world_size, device=device
        )
    else:
        # Standard tensor for other operations
        x = torch.tensor([rank + 1.0, rank + 2.0, rank + 3.0, rank + 4.0], device=device)

    logger.debug(f"Rank {rank}: Initial tensor for {func_name}: {x}")

    # Create a wrapper function that applies the collective and post-processing
    def wrapper(tensor):
        result = tensor.clone()
        # Apply the collective operation
        collective_result = collective_fn(result, device_mesh)
        # Add post-processing operations to test graph continuity
        collective_result = collective_result + 1.0
        collective_result = collective_result * 2.0
        return collective_result

    try:
        # Test eager execution
        logger.debug(f"Rank {rank}: Testing eager {func_name}...")
        eager_result = wrapper(x)
        logger.debug(f"Rank {rank}: Eager {func_name} result: {eager_result}")

        # Test torch.compile
        logger.debug(f"Rank {rank}: Testing compiled {func_name}...")
        compiled_func = torch.compile(wrapper, backend="neuron")
        compiled_result = compiled_func(x)
        logger.debug(f"Rank {rank}: Compiled {func_name} result: {compiled_result}")

        # Verify results match
        if torch.allclose(eager_result, compiled_result, atol=1e-5):
            logger.debug(
                f"Rank {rank}: {func_name} results match - functional collective "
                f"works with torch.compile!"
            )
            return "SUCCESS"
        else:
            logger.error(f"Rank {rank}: {func_name} results don't match")
            logger.debug(f"Rank {rank}: Eager: {eager_result}")
            logger.debug(f"Rank {rank}: Compiled: {compiled_result}")
            return "MISMATCH"

    except Exception as e:
        logger.error(f"Rank {rank}: {func_name} test failed: {e}")
        import traceback

        traceback.print_exc()
        return f"FAILED: {e!s}"


def run_functional_collectives_test(rank, world_size, kwargs):
    """Test all relevant functional collective operations with torch.compile"""

    logger.debug(f"=== Rank {rank}/{world_size}: Testing Functional Collective Operations ===")
    logger.debug("Using backend: neuron_backend")

    # List the operations we're testing
    if rank == 0:
        logger.debug("Testing the following functional collective operations:")
        for i, (op_name, _, _, _) in enumerate(COLLECTIVE_OPERATIONS, 1):
            logger.debug(f"  {i:2d}. {op_name}")

    device = torch.device(f"neuron:{rank}")
    # Create device mesh like TorchTitan does
    device_mesh = DeviceMesh("neuron", list(range(world_size)))
    logger.debug(f"Rank {rank}: Created device mesh: {device_mesh}, using device: {device}")

    # Test results tracking
    results = {}

    # Test each functional collective
    for collective_config in COLLECTIVE_OPERATIONS:
        func_name = collective_config[0]
        try:
            result = _run_single_collective(
                collective_config, rank, world_size, device_mesh, device
            )
            results[func_name] = result

            # Synchronize between tests
            dist.barrier()

        except Exception as e:
            logger.error(f"Rank {rank}: Failed to test {func_name}: {e}")
            results[func_name] = f"SETUP_FAILED: {e!s}"

    # Print summary
    if rank == 0:
        logger.info("\n" + "=" * 80)
        logger.info("FUNCTIONAL COLLECTIVES + TORCH.COMPILE TEST SUMMARY")
        logger.info("=" * 80)

        success_count = 0
        total_count = len(results)

        for func_name, status in results.items():
            logger.info(f"{func_name:35} | {status}")
            if status == "SUCCESS":
                success_count += 1

        logger.info("=" * 80)
        logger.info(
            f"SUCCESS RATE: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)"
        )
        logger.info("=" * 80)

    # Note: Don't call dist.destroy_process_group() here - DistributedTester handles cleanup
    logger.info(f"Rank {rank}: Test complete")


class TestFunctionalCollectives(BaseCollectiveOpTest):
    """Test class for functional collectives component tests using DistributedTester."""

    @pytest.mark.multi_device
    def test_functional_collectives(self):
        """Test functional collective operations with torch.compile."""
        self.distributed_tester.run_test(run_functional_collectives_test)


if __name__ == "__main__":
    import pytest

    pytest.main(["-vs", __file__])
