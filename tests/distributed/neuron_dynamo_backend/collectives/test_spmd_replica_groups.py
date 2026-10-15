"""
Test that collective legalization produces SPMD-compatible replica groups.

For multi-dimensional parallelism (TP+FSDP), all ranks must produce identical
replica groups so they compile to the same NEFF. This test calls the actual
`_get_process_group_ranks` function and verifies SPMD compatibility.

This test should FAIL with the old implementation and PASS after the fix.
"""

import pytest
from torch.distributed.device_mesh import init_device_mesh

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from tests.distributed.utils import broadcast_object_from_rank0
from torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization import (
    _get_process_group_ranks,
)

pytestmark = pytest.mark.neuron_cores(8)


def run_spmd_replica_groups_test(rank, world_size, kwargs):
    """
    Test that _get_process_group_ranks returns SPMD-compatible results.

    For SPMD, all ranks must get identical replica groups. The OLD implementation
    fails this because dist.get_process_group_ranks() only returns the calling
    rank's group. The NEW implementation returns all replica groups.
    """
    # Create 2D mesh: outer=FSDP, inner=TP
    tp_size = 2
    fsdp_size = world_size // tp_size
    mesh = init_device_mesh("neuron", (fsdp_size, tp_size), mesh_dim_names=("fsdp", "tp"))

    # Get PG names for each dimension
    fsdp_pg_name = mesh._dim_group_names[0]
    tp_pg_name = mesh._dim_group_names[1]

    # Call the actual function under test
    # With the fix, this returns list[list[int]] - all replica groups
    fsdp_result = _get_process_group_ranks(fsdp_pg_name)
    tp_result = _get_process_group_ranks(tp_pg_name)

    # Broadcast rank 0's results to all ranks (host-side; see helper docstring)
    rank0_fsdp = [broadcast_object_from_rank0(fsdp_result, rank, world_size, tag="fsdp")]
    rank0_tp = [broadcast_object_from_rank0(tp_result, rank, world_size, tag="tp")]

    # SPMD requirement: all ranks must produce identical results
    assert (
        fsdp_result == rank0_fsdp[0]
    ), f"FSDP groups not SPMD: rank 0={rank0_fsdp[0]}, rank {rank}={fsdp_result}"
    assert (
        tp_result == rank0_tp[0]
    ), f"TP groups not SPMD: rank 0={rank0_tp[0]}, rank {rank}={tp_result}"

    # Verify the result is a list of lists (full replica groups)
    assert isinstance(fsdp_result, list), f"Expected list, got {type(fsdp_result)}"
    assert isinstance(tp_result, list), f"Expected list, got {type(tp_result)}"
    assert all(isinstance(g, list) for g in fsdp_result), "FSDP result should be list of lists"
    assert all(isinstance(g, list) for g in tp_result), "TP result should be list of lists"

    # Verify expected structure for 2D mesh
    # TP (inner dim): should have fsdp_size groups, each with tp_size ranks
    assert len(tp_result) == fsdp_size, f"Expected {fsdp_size} TP groups, got {len(tp_result)}"
    # FSDP (outer dim): should have tp_size groups, each with fsdp_size ranks
    assert len(fsdp_result) == tp_size, f"Expected {tp_size} FSDP groups, got {len(fsdp_result)}"


def _verify_spmd_and_structure(
    rank, world_size, results, expected_num_groups, expected_group_size, dim_name
):
    """Helper to verify SPMD and structure for a dimension."""
    result = results[dim_name]

    # Broadcast rank 0's result (host-side; see helper docstring)
    rank0_result = [broadcast_object_from_rank0(result, rank, world_size, tag=f"verify_{dim_name}")]

    # SPMD check
    assert result == rank0_result[0], f"{dim_name} not SPMD: rank 0 vs rank {rank}"

    # Structure check
    assert isinstance(result, list) and all(isinstance(g, list) for g in result)
    assert (
        len(result) == expected_num_groups
    ), f"{dim_name}: expected {expected_num_groups} groups, got {len(result)}"
    for g in result:
        assert (
            len(g) == expected_group_size
        ), f"{dim_name}: expected group size {expected_group_size}, got {len(g)}"


def run_3d_mesh_spmd_test(rank, world_size, kwargs):
    """Test 3D mesh (dp=2, fsdp=2, tp=2)."""
    mesh = init_device_mesh("neuron", (2, 2, 2), mesh_dim_names=("dp", "fsdp", "tp"))
    results = {
        name: _get_process_group_ranks(mesh._dim_group_names[i])
        for i, name in enumerate(["dp", "fsdp", "tp"])
    }

    # dp (dim 0): 2*2=4 groups of 2 ranks
    _verify_spmd_and_structure(rank, world_size, results, 4, 2, "dp")
    # fsdp (dim 1): 2*2=4 groups of 2 ranks
    _verify_spmd_and_structure(rank, world_size, results, 4, 2, "fsdp")
    # tp (dim 2): 2*2=4 groups of 2 ranks
    _verify_spmd_and_structure(rank, world_size, results, 4, 2, "tp")


class TestSPMDReplicaGroups(BaseCollectiveOpTest):
    """Test SPMD replica group derivation for multi-dimensional meshes."""

    @property
    def world_size(self):
        return 8

    @pytest.mark.multi_device
    def test_2d_mesh_spmd_replica_groups(self):
        """Verify 2D mesh (fsdp=4, tp=2) returns identical results on all ranks."""
        self.distributed_tester.run_test(run_spmd_replica_groups_test)

    @pytest.mark.multi_device
    def test_3d_mesh_spmd_replica_groups(self):
        """Verify 3D mesh (dp=2, fsdp=2, tp=2) produces correct SPMD replica groups."""
        self.distributed_tester.run_test(run_3d_mesh_spmd_test)
