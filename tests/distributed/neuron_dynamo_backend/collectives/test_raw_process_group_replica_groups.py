"""Distributed coverage for raw new_group replica-group reconstruction."""

from unittest.mock import patch

import pytest
import torch.distributed as dist

from tests.distributed.collective_ops.base_collective_op import BaseCollectiveOpTest
from torch_neuronx.distributed import mesh_registry
from torch_neuronx.distributed.mesh_registry import (
    clear_mesh_registry,
    get_all_replica_groups,
)


def _get_member_groups(rank, groups):
    member_groups = None
    for ranks in groups:
        device_group = dist.new_group(ranks=ranks)
        cpu_group = dist.new_group(ranks=ranks, backend="gloo")
        if rank in ranks:
            member_groups = (device_group, cpu_group)
    assert member_groups is not None
    return member_groups


def run_raw_partition_test(rank, world_size, kwargs):
    """Every raw group name resolves to its complete sibling partition."""
    clear_mesh_registry()

    tp_groups = [[0, 1], [2, 3]]
    fsdp_groups = [[0, 2], [1, 3]]

    tp_member_groups = _get_member_groups(rank, tp_groups)
    fsdp_member_groups = _get_member_groups(rank, fsdp_groups)

    for group in tp_member_groups:
        assert get_all_replica_groups(group.group_name) == tp_groups
    for group in fsdp_member_groups:
        assert get_all_replica_groups(group.group_name) == fsdp_groups


def run_incomplete_partition_test(rank, world_size, kwargs):
    """A raw partial group errors (loud) and falls back to its local rank cell."""
    clear_mesh_registry()
    group = dist.new_group(ranks=[0, 1])

    if rank in [0, 1]:
        with patch.object(mesh_registry.logger, "error") as mock_error:
            result = get_all_replica_groups(group.group_name)

        assert result == [[0, 1]]
        mock_error.assert_called_once_with(
            f"No registered replica groups for PG '{group.group_name}' in SPMD mode "
            "(local ranks [0, 1], world 4); falling back to per-rank groups, "
            "which violates the SPMD identical-replica-groups invariant and may give wrong "
            "collectives. Register the mesh or set TORCH_NEURONX_SPMD_DISABLE=1 to opt in."
        )


class TestRawProcessGroupReplicaGroups(BaseCollectiveOpTest):
    """Verify PTN's raw new_group capture on a four-rank partition."""

    @property
    def world_size(self):
        return 4

    @pytest.mark.multi_device
    def test_raw_partition_replica_groups(self):
        self.distributed_tester.run_test(run_raw_partition_test)

    @pytest.mark.multi_device
    def test_incomplete_partition_errors_and_falls_back(self):
        self.distributed_tester.run_test(run_incomplete_partition_test)
