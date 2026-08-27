"""
Unit tests for collective_legalization fx pass.
"""

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch.fx import GraphModule, Node

from torch_neuronx.distributed import mesh_registry
from torch_neuronx.neuron_dynamo_backend.fx.passes import collective_legalization
from torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization import (
    CollectiveLegalization,
)


def _process_group(group_name):
    return MagicMock(spec=dist.ProcessGroup, group_name=group_name)


class TestGetProcessGroupRanks:
    """Test get_process_group_ranks function"""

    @patch(
        "torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization.get_all_replica_groups"
    )
    def test_get_process_group_ranks_uses_registry(self, mock_get_all):
        """Test that _get_process_group_ranks delegates to get_all_replica_groups"""
        mock_get_all.return_value = [[0, 1], [2, 3]]

        result = collective_legalization._get_process_group_ranks("test_group")

        mock_get_all.assert_called_once_with("test_group")
        assert result == [[0, 1], [2, 3]]

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=4)
    @patch(
        "torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization.get_all_replica_groups"
    )
    def test_get_process_group_ranks_unresolved_falls_back_to_world(
        self, mock_get_all, mock_world_size, mock_is_initialized
    ):
        """A failed lookup degrades to the full world group instead of aborting."""
        mock_get_all.side_effect = RuntimeError("unresolved group")

        result = collective_legalization._get_process_group_ranks("invalid_group")

        assert result == [[0, 1, 2, 3]]

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch(
        "torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization.get_all_replica_groups"
    )
    def test_get_process_group_ranks_fallback_not_initialized(
        self, mock_get_all, mock_is_initialized
    ):
        """When dist is not initialized, the unresolved fallback uses world_size=1."""
        mock_get_all.side_effect = RuntimeError("unresolved group")

        result = collective_legalization._get_process_group_ranks("invalid_group")

        assert result == [[0]]


class TestNeuronSpmdDisable:
    """Test TORCH_NEURONX_SPMD_DISABLE env var bypasses the mesh registry."""

    @patch("torch.distributed.get_process_group_ranks", return_value=[0, 1])
    @patch("torch.distributed.get_world_size", return_value=4)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.distributed_c10d._resolve_process_group")
    def test_disable_bypasses_registry(
        self, mock_resolve, mock_is_initialized, mock_world_size, mock_ranks
    ):
        """When TORCH_NEURONX_SPMD_DISABLE is set, get_all_replica_groups falls back to per-rank."""
        from torch_neuronx.distributed.mesh_registry import (
            _MESH_REGISTRY,
            get_all_replica_groups,
        )

        _MESH_REGISTRY["test_pg"] = [[0, 1], [2, 3]]
        try:
            os.environ["TORCH_NEURONX_SPMD_DISABLE"] = "1"
            result = get_all_replica_groups("test_pg")
            assert result == [[0, 1]], f"Expected fallback, got {result}"
            mock_resolve.assert_called_once_with("test_pg")
        finally:
            os.environ.pop("TORCH_NEURONX_SPMD_DISABLE", None)
            _MESH_REGISTRY.pop("test_pg", None)

    def test_registry_used_when_not_disabled(self):
        """When TORCH_NEURONX_SPMD_DISABLE is unset, registry is used."""
        from torch_neuronx.distributed.mesh_registry import (
            _MESH_REGISTRY,
            get_all_replica_groups,
        )

        _MESH_REGISTRY["test_pg2"] = [[0, 1], [2, 3]]
        os.environ.pop("TORCH_NEURONX_SPMD_DISABLE", None)
        try:
            result = get_all_replica_groups("test_pg2")
            assert result == [[0, 1], [2, 3]]
        finally:
            _MESH_REGISTRY.pop("test_pg2", None)


class TestRawNewGroupCapture:
    """Test raw new_group capture without requiring Neuron hardware."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        mesh_registry.clear_mesh_registry()
        yield
        mesh_registry.clear_mesh_registry()

    @patch.object(mesh_registry, "_original_new_group")
    @patch.object(mesh_registry.dist, "get_world_size", return_value=4)
    def test_registers_complete_partition(self, mock_world_size, mock_new_group):
        pg01 = _process_group("pg_01")
        pg23 = _process_group("pg_23")
        mock_new_group.side_effect = [pg01, pg23]

        mesh_registry._patched_new_group(ranks=[0, 1])
        mesh_registry._patched_new_group(ranks=[2, 3])

        expected = [[0, 1], [2, 3]]
        assert mesh_registry.get_all_replica_groups("pg_01") == expected
        assert mesh_registry.get_all_replica_groups("pg_23") == expected

    @patch.object(mesh_registry, "_original_new_group")
    @patch.object(mesh_registry.dist, "get_world_size", return_value=4)
    def test_nonmember_call_still_completes_partition(self, mock_world_size, mock_new_group):
        pg01 = _process_group("pg_01")
        mock_new_group.side_effect = [pg01, mesh_registry.dist.GroupMember.NON_GROUP_MEMBER]

        mesh_registry._patched_new_group(ranks=[0, 1])
        mesh_registry._patched_new_group(ranks=[2, 3])

        assert mesh_registry.get_all_replica_groups("pg_01") == [[0, 1], [2, 3]]

    @patch.object(mesh_registry, "_original_new_group")
    @patch.object(mesh_registry.dist, "get_world_size", return_value=4)
    def test_incomplete_partition_errors_and_uses_local_ranks(
        self, mock_world_size, mock_new_group
    ):
        pg23 = _process_group("pg_23")
        mock_new_group.return_value = pg23
        mesh_registry._patched_new_group(ranks=[2, 3])

        with (
            patch.object(
                mesh_registry.dist.distributed_c10d,
                "_resolve_process_group",
                return_value=pg23,
            ),
            patch.object(mesh_registry.dist, "is_initialized", return_value=True),
            patch.object(mesh_registry.dist, "get_process_group_ranks", return_value=[2, 3]),
            patch.object(mesh_registry.logger, "error") as mock_error,
        ):
            result = mesh_registry.get_all_replica_groups("pg_23")

        assert result == [[2, 3]]
        mock_error.assert_called_once_with(
            "No registered replica groups for PG 'pg_23' in SPMD mode "
            "(local ranks [2, 3], world 4); falling back to per-rank groups, "
            "which violates the SPMD identical-replica-groups invariant and may give wrong "
            "collectives. Register the mesh or set TORCH_NEURONX_SPMD_DISABLE=1 to opt in."
        )

    @patch.object(mesh_registry, "_original_new_group")
    @patch.object(mesh_registry.dist, "get_world_size", return_value=4)
    def test_incomplete_partition_does_not_poison_later_partition(
        self, mock_world_size, mock_new_group
    ):
        pg_invalid = _process_group("pg_invalid")
        pg01 = _process_group("pg_01")
        pg_overlap = _process_group("pg_overlap")
        pg23 = _process_group("pg_23")
        mock_new_group.side_effect = [pg_invalid, pg_overlap, pg01, pg23]

        mesh_registry._patched_new_group(ranks=[0, 1])
        mesh_registry._patched_new_group(ranks=[1, 2])
        mesh_registry._patched_new_group(ranks=[0, 1])
        mesh_registry._patched_new_group(ranks=[2, 3])

        assert mesh_registry.get_all_replica_groups("pg_01") == [[0, 1], [2, 3]]
        with (
            patch.object(
                mesh_registry.dist.distributed_c10d,
                "_resolve_process_group",
                return_value=pg_invalid,
            ),
            patch.object(mesh_registry.dist, "is_initialized", return_value=True),
            patch.object(mesh_registry.dist, "get_process_group_ranks", return_value=[1, 2]),
            patch.object(mesh_registry.logger, "error") as mock_error,
        ):
            result = mesh_registry.get_all_replica_groups("pg_invalid")

        assert result == [[1, 2]]
        mock_error.assert_called_once_with(
            "No registered replica groups for PG 'pg_invalid' in SPMD mode "
            "(local ranks [1, 2], world 4); falling back to per-rank groups, "
            "which violates the SPMD identical-replica-groups invariant and may give wrong "
            "collectives. Register the mesh or set TORCH_NEURONX_SPMD_DISABLE=1 to opt in."
        )

    @pytest.mark.parametrize("member_cell", ["a", "b"])
    def test_interleaved_device_and_cpu_partitions(self, member_cell):
        """vLLM's device/CPU group order closes both independent partitions."""
        nonmember = mesh_registry.dist.GroupMember.NON_GROUP_MEMBER
        if member_cell == "a":
            process_groups = [
                _process_group("device_a"),
                _process_group("cpu_a"),
                nonmember,
                nonmember,
            ]
            local_names = ["device_a", "cpu_a"]
        else:
            process_groups = [
                nonmember,
                nonmember,
                _process_group("device_b"),
                _process_group("cpu_b"),
            ]
            local_names = ["device_b", "cpu_b"]

        with (
            patch.object(mesh_registry, "_original_new_group", side_effect=process_groups),
            patch.object(mesh_registry.dist, "get_world_size", return_value=4),
        ):
            mesh_registry._patched_new_group([0, 1])
            mesh_registry._patched_new_group([0, 1], backend="gloo")
            mesh_registry._patched_new_group([2, 3])
            mesh_registry._patched_new_group([2, 3], backend="gloo")

        expected = [[0, 1], [2, 3]]
        for local_name in local_names:
            assert mesh_registry.get_all_replica_groups(local_name) == expected
        assert mesh_registry._RAW_PENDING == []

    @patch.object(mesh_registry, "_original_new_group")
    @patch.object(mesh_registry.dist, "get_world_size", return_value=4)
    def test_use_local_synchronization_member_only_sequence_falls_back(
        self, mock_world_size, mock_new_group
    ):
        """With use_local_synchronization=True a rank only calls new_group for the
        groups it belongs to, so the partition never tiles the world and lookup
        falls back to local ranks instead of fabricating a full partition."""
        pg01 = _process_group("pg_01")
        mock_new_group.return_value = pg01

        mesh_registry._patched_new_group(ranks=[0, 1], use_local_synchronization=True)

        # The flag reaches torch unchanged.
        assert mock_new_group.call_args.kwargs["use_local_synchronization"] is True

        with (
            patch.object(
                mesh_registry.dist.distributed_c10d,
                "_resolve_process_group",
                return_value=pg01,
            ),
            patch.object(mesh_registry.dist, "is_initialized", return_value=True),
            patch.object(mesh_registry.dist, "get_process_group_ranks", return_value=[0, 1]),
            patch.object(mesh_registry.logger, "error") as mock_error,
        ):
            result = mesh_registry.get_all_replica_groups("pg_01")

        assert result == [[0, 1]]
        mock_error.assert_called_once()

    def test_registry_lookup_returns_defensive_copy(self):
        mesh_registry._register_replica_groups("pg", [[0, 1], [2, 3]])

        result = mesh_registry.get_all_replica_groups("pg")
        result[0].append(99)

        assert mesh_registry.get_all_replica_groups("pg") == [[0, 1], [2, 3]]


def _new_group_positional_args(**overrides):
    """Build a positional arg tuple from torch's ``new_group`` signature.

    Derived rather than hand-written so a torch parameter reorder changes the
    positions under test instead of leaving a stale hardcoded tuple that keeps
    passing while production reads the wrong argument.

    Reads the unpatched function: importing torch_neuronx rebinds
    ``dist.new_group`` to a ``*args, **kwargs`` wrapper whose signature exposes
    no parameter names.
    """
    parameters = inspect.signature(mesh_registry._original_new_group).parameters
    names = list(parameters)
    last = max(names.index(name) for name in overrides)
    return tuple(overrides.get(name, parameters[name].default) for name in names[: last + 1])


class TestNormalizeNewGroupRanks:
    """Directly exercise ``_normalize_new_group_ranks`` argument extraction."""

    def test_sort_ranks_resolves_by_name_not_position(self):
        """``sort_ranks`` binds by parameter name, wherever torch places it."""
        args = _new_group_positional_args(ranks=[3, 1, 2], sort_ranks=True)
        result = mesh_registry._normalize_new_group_ranks(args, {}, world_size=4)
        assert result == (1, 2, 3)

    def test_sort_ranks_false_positional_preserves_order(self):
        """``sort_ranks=False`` (positional) keeps the caller's rank order."""
        args = _new_group_positional_args(ranks=[3, 1, 2], sort_ranks=False)
        result = mesh_registry._normalize_new_group_ranks(args, {}, world_size=4)
        assert result == (3, 1, 2)

    def test_positional_sort_ranks_matches_keyword_form(self):
        """The positional and keyword spellings of one call agree.

        A position-based read would diverge from the keyword read as soon as
        torch inserts a parameter ahead of ``sort_ranks``.
        """
        positional = mesh_registry._normalize_new_group_ranks(
            _new_group_positional_args(ranks=[3, 1, 2], sort_ranks=False), {}, world_size=4
        )
        keyword = mesh_registry._normalize_new_group_ranks(
            (), {"ranks": [3, 1, 2], "sort_ranks": False}, world_size=4
        )
        assert positional == keyword == (3, 1, 2)

    def test_sort_ranks_false_keyword_preserves_order(self):
        """``sort_ranks=False`` (keyword) keeps the caller's rank order."""
        result = mesh_registry._normalize_new_group_ranks(
            (), {"ranks": [3, 1, 2], "sort_ranks": False}, world_size=4
        )
        assert result == (3, 1, 2)

    def test_ranks_none_defaults_to_world(self):
        """``ranks=None`` normalizes to the whole world."""
        result = mesh_registry._normalize_new_group_ranks((), {}, world_size=4)
        assert result == (0, 1, 2, 3)


class TestRegisterReplicaGroupsConflict:
    """Cover the conflict-raise in ``_register_replica_groups`` and its callers."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        mesh_registry.clear_mesh_registry()
        yield
        mesh_registry.clear_mesh_registry()

    def test_conflicting_registration_raises(self):
        """Re-registering a name with different groups raises RuntimeError."""
        mesh_registry._register_replica_groups("pg", [[0, 1], [2, 3]])
        with pytest.raises(RuntimeError, match="already registered with different"):
            mesh_registry._register_replica_groups("pg", [[0, 2], [1, 3]])

    def test_identical_reregistration_is_idempotent(self):
        """Re-registering the same groups is a no-op, not a raise."""
        mesh_registry._register_replica_groups("pg", [[0, 1], [2, 3]])
        mesh_registry._register_replica_groups("pg", [[0, 1], [2, 3]])
        assert mesh_registry.get_all_replica_groups("pg") == [[0, 1], [2, 3]]

    @patch.object(mesh_registry, "_original_new_group")
    @patch.object(mesh_registry.dist, "get_world_size", return_value=4)
    def test_capture_swallows_conflict_and_keeps_first(self, mock_world_size, mock_new_group):
        """A conflicting capture is swallowed by ``_patched_new_group``; the
        first-registered table survives and the second name goes unregistered."""
        pg01 = _process_group("shared_name")
        pg23 = _process_group("shared_name")
        mock_new_group.side_effect = [pg01, pg23]

        # First partition [[0,1],[2,3]] registers "shared_name" -> that table.
        mesh_registry._patched_new_group(ranks=[0, 1])
        mesh_registry._patched_new_group(ranks=[2, 3])
        assert mesh_registry.get_all_replica_groups("shared_name") == [[0, 1], [2, 3]]

        # A second, differently-shaped partition reusing the same name conflicts;
        # the raise is swallowed and the original table is preserved (not corrupted).
        pg02 = _process_group("shared_name")
        pg13 = _process_group("shared_name")
        mock_new_group.side_effect = [pg02, pg13]
        mesh_registry._patched_new_group(ranks=[0, 2])
        mesh_registry._patched_new_group(ranks=[1, 3])

        assert mesh_registry.get_all_replica_groups("shared_name") == [[0, 1], [2, 3]]


class TestCollectiveLegalization:
    """Test CollectiveLegalization pass"""

    def create_mock_fx_node(self, op_name, args):
        """Helper to create a mock FX node"""
        node = MagicMock(spec=Node)
        node.op = "call_function"
        node.target = op_name
        node.args = args
        return node

    def create_mock_graph_module(self, nodes):
        """Helper to create a mock GraphModule with given nodes"""
        gm = MagicMock(spec=GraphModule)
        gm.graph = MagicMock()
        gm.graph.nodes = nodes
        gm.recompile = MagicMock()
        return gm

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_all_reduce_legalization(self, mock_get_ranks):
        """Test all_reduce operation legalization"""
        mock_get_ranks.return_value = [[0, 1, 2, 3]]

        # Create mock node for all_reduce operation
        node = self.create_mock_fx_node(
            "_c10d_functional.all_reduce.default", ("tensor", "'sum'", "'test_group'")
        )

        gm = self.create_mock_graph_module([node])

        # Apply legalization
        pass_instance = CollectiveLegalization()
        result = pass_instance(gm)

        # Verify the group argument was transformed
        assert node.args[2] == "[[0, 1, 2, 3]]"
        gm.recompile.assert_called_once()
        assert result.modified is True

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_all_gather_legalization(self, mock_get_ranks):
        """Test all_gather_into_tensor operation legalization"""
        mock_get_ranks.return_value = [[0, 2, 4, 6]]

        # Create mock node for all_gather_into_tensor operation
        node = self.create_mock_fx_node(
            "_c10d_functional.all_gather_into_tensor.default",
            ("input_tensor", "output_tensor", "test_group_2"),
        )

        gm = self.create_mock_graph_module([node])

        # Apply legalization
        pass_instance = CollectiveLegalization()
        pass_instance(gm)

        # Verify the group argument was transformed (group_index=2 for all_gather)
        assert node.args[2] == "[[0, 2, 4, 6]]"

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_reduce_scatter_legalization(self, mock_get_ranks):
        """Test reduce_scatter_tensor operation legalization"""
        mock_get_ranks.return_value = [[1, 3, 5, 7]]

        # Create mock node for reduce_scatter_tensor operation
        node = self.create_mock_fx_node(
            "_c10d_functional.reduce_scatter_tensor.default",
            ("input_tensor", "'sum'", "scatter_list", "scatter_group"),
        )

        gm = self.create_mock_graph_module([node])

        # Apply legalization
        pass_instance = CollectiveLegalization()
        pass_instance(gm)

        # Verify the group argument was transformed (group_index=3 for reduce_scatter)
        assert node.args[3] == "[[1, 3, 5, 7]]"

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_all_to_all_legalization(self, mock_get_ranks):
        """Test all_to_all_single operation legalization"""
        mock_get_ranks.return_value = [[0, 1]]

        # Create mock node for all_to_all_single operation
        node = self.create_mock_fx_node(
            "_c10d_functional.all_to_all_single.default",
            ("output", "input", "output_split_sizes", "all_to_all_group"),
        )

        gm = self.create_mock_graph_module([node])

        # Apply legalization
        pass_instance = CollectiveLegalization()
        pass_instance(gm)

        # Verify the group argument was transformed (group_index=3 for all_to_all)
        assert node.args[3] == "[[0, 1]]"

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_broadcast_legalization(self, mock_get_ranks):
        """broadcast legalizes its group arg like any other collective; src is untouched."""
        mock_get_ranks.return_value = [[0, 1], [2, 3]]

        # broadcast(tensor, src, group): group is at index 2, src at index 1
        node = self.create_mock_fx_node(
            "_c10d_functional.broadcast.default", ("tensor", 1, "broadcast_group")
        )

        gm = self.create_mock_graph_module([node])

        pass_instance = CollectiveLegalization()
        result = pass_instance(gm)

        assert node.args[1] == 1  # src unchanged
        assert node.args[2] == "[[0, 1], [2, 3]]"
        gm.recompile.assert_called_once()
        assert result.modified is True

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_shard_dim_alltoall_legalization(self, mock_get_ranks):
        """Test shard_dim_alltoall operation legalization"""
        mock_get_ranks.return_value = [[0, 1]]

        node = self.create_mock_fx_node(
            "_dtensor.shard_dim_alltoall.default",
            ("input_tensor", "gather_dim", "shard_dim", "shard_dim_alltoall_group"),
        )

        gm = self.create_mock_graph_module([node])

        pass_instance = CollectiveLegalization()
        pass_instance(gm)

        assert node.args[3] == "[[0, 1]]"

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_multiple_collective_ops_legalization(self, mock_get_ranks):
        """Test legalization of multiple collective operations"""
        # Mock different return values for different calls
        mock_get_ranks.side_effect = [[[0, 1, 2, 3]], [[0, 1]], [[2, 3]]]

        # Create multiple collective operation nodes
        all_reduce_node = self.create_mock_fx_node(
            "_c10d_functional.all_reduce.default", ("tensor1", "'sum'", "group1")
        )
        all_gather_node = self.create_mock_fx_node(
            "_c10d_functional.all_gather_into_tensor.default", ("tensor2", "output2", "group2")
        )
        reduce_scatter_node = self.create_mock_fx_node(
            "_c10d_functional.reduce_scatter_tensor.default",
            ("tensor3", "'sum'", "scatter_list", "group3"),
        )

        gm = self.create_mock_graph_module([all_reduce_node, all_gather_node, reduce_scatter_node])

        # Apply legalization
        pass_instance = CollectiveLegalization()
        pass_instance(gm)

        # Verify all operations were transformed
        assert all_reduce_node.args[2] == "[[0, 1, 2, 3]]"
        assert all_gather_node.args[2] == "[[0, 1]]"
        assert reduce_scatter_node.args[3] == "[[2, 3]]"

    def test_non_collective_ops_ignored(self):
        """Test that non-collective operations are ignored"""
        # Create mock nodes for non-collective operations
        regular_node = self.create_mock_fx_node("aten.add.default", ("tensor1", "tensor2"))

        function_node = MagicMock()
        function_node.op = "call_method"  # Different op type
        function_node.target = "add"

        gm = self.create_mock_graph_module([regular_node, function_node])

        # Store original args
        original_args = regular_node.args

        # Apply legalization
        pass_instance = CollectiveLegalization()
        result = pass_instance(gm)

        # Verify non-collective nodes weren't modified
        assert regular_node.args == original_args
        assert result.modified is False

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_unknown_collective_op_ignored(self, mock_get_ranks):
        """Test that unknown collective operations are ignored"""
        # Create mock node for unknown collective operation
        node = self.create_mock_fx_node(
            "_c10d_functional.unknown_collective.default", ("tensor", "group")
        )

        gm = self.create_mock_graph_module([node])
        original_args = node.args

        # Apply legalization
        pass_instance = CollectiveLegalization()
        result = pass_instance(gm)

        # Verify unknown operation wasn't modified
        assert node.args == original_args
        mock_get_ranks.assert_not_called()
        assert result.modified is False

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_args_modification_preserves_other_args(self, mock_get_ranks):
        """Test that only the group argument is modified, others preserved"""
        mock_get_ranks.return_value = [[0, 1]]

        # Create node with multiple arguments
        original_args = ("input_tensor", "'sum'", "original_group", "extra_arg")
        node = self.create_mock_fx_node("_c10d_functional.all_reduce.default", original_args)

        gm = self.create_mock_graph_module([node])

        # Apply legalization
        pass_instance = CollectiveLegalization()
        pass_instance(gm)

        # Verify only group argument (index 2) was modified
        assert node.args[0] == "input_tensor"  # Unchanged
        assert node.args[1] == "'sum'"  # Unchanged
        assert node.args[2] == "[[0, 1]]"  # Changed
        assert node.args[3] == "extra_arg"  # Unchanged


class TestCollectiveTransformsIntegration:
    """Integration tests for collective transforms functionality"""

    def create_mock_fx_node(self, op_name, args):
        """Helper to create a mock FX node"""
        node = MagicMock()
        node.op = "call_function"
        node.target = op_name
        node.args = args
        return node

    def create_mock_graph_module(self, nodes):
        """Helper to create a mock GraphModule with given nodes"""
        gm = MagicMock(spec=GraphModule)
        gm.graph = MagicMock()
        gm.graph.nodes = nodes
        gm.recompile = MagicMock()
        return gm

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_end_to_end_transformation(self, mock_get_ranks):
        """Test complete transformation workflow"""

        # Set up mock to return different ranks for different groups
        def mock_ranks_side_effect(group_name):
            group_mapping = {
                "group_0": [[0, 1, 2, 3]],
                "group_1": [[0, 1]],
                "group_2": [[2, 3]],
            }
            return group_mapping.get(group_name, [[0]])

        mock_get_ranks.side_effect = mock_ranks_side_effect

        # Create a realistic scenario with multiple collective operations
        all_reduce_node = self.create_mock_fx_node(
            "_c10d_functional.all_reduce.default", ("gradients", "'sum'", "group_0")
        )

        all_gather_node = self.create_mock_fx_node(
            "_c10d_functional.all_gather_into_tensor.default",
            ("local_tensor", "gathered_tensor", "group_1"),
        )

        reduce_scatter_node = self.create_mock_fx_node(
            "_c10d_functional.reduce_scatter_tensor.default",
            ("input", "'sum'", "scatter_list", "group_2"),
        )

        # Add some non-collective operations
        regular_node = self.create_mock_fx_node("aten.relu.default", ("input_tensor",))

        gm = self.create_mock_graph_module(
            [regular_node, all_reduce_node, all_gather_node, reduce_scatter_node]
        )

        # Apply transformation
        pass_instance = CollectiveLegalization()
        result = pass_instance(gm)

        # Verify transformations
        assert all_reduce_node.args[2] == "[[0, 1, 2, 3]]"
        assert all_gather_node.args[2] == "[[0, 1]]"
        assert reduce_scatter_node.args[3] == "[[2, 3]]"

        # Verify non-collective operation unchanged
        assert regular_node.args == ("input_tensor",)

        # Verify graph was recompiled
        gm.recompile.assert_called_once()
        assert result.modified is True


class TestCollectiveOpsMapping:
    """Test the collective operations to group index mapping"""

    @patch.object(collective_legalization, "_get_process_group_ranks")
    def test_all_supported_ops_are_legalized(self, mock_get_ranks):
        """Test that all expected collective operations are handled"""
        mock_get_ranks.return_value = [[0, 1]]

        # Test each supported op
        supported_ops = [
            ("_c10d_functional.all_reduce.default", 2),
            ("_c10d_functional.all_gather_into_tensor.default", 2),
            ("_c10d_functional.reduce_scatter_tensor.default", 3),
            ("_c10d_functional.all_to_all_single.default", 3),
            ("_dtensor.shard_dim_alltoall.default", 3),
            ("_c10d_functional.broadcast.default", 2),
        ]

        for op_name, group_index in supported_ops:
            args = ["arg"] * (group_index + 1)
            args[group_index] = "test_group"

            node = MagicMock()
            node.op = "call_function"
            node.target = op_name
            node.args = tuple(args)

            gm = MagicMock(spec=GraphModule)
            gm.graph.nodes = [node]

            pass_instance = CollectiveLegalization()
            result = pass_instance(gm)

            assert node.args[group_index] == "[[0, 1]]", f"Failed for {op_name}"
            assert result.modified is True


class TestBroadcastLowering:
    """End-to-end FX -> StableHLO lowering of broadcast.

    Confirms the group arg controls the emitted replica_groups: an
    un-legalized name falls back to the whole world, while a legalized
    subgroup lowers to that subgroup.
    """

    def _broadcast_stablehlo(self, group_str, src, world_size=4):
        import torch.distributed._functional_collectives
        from torch.fx.experimental.proxy_tensor import make_fx

        from torch_neuronx.neuron_dynamo_backend.fx.fx_transform import (
            convert_fx_to_stablehlo,
        )

        def fn(x):
            y = torch.ops._c10d_functional.broadcast.default(x, src, group_str)
            return torch.ops._c10d_functional.wait_tensor.default(y)

        example = torch.randn(8, dtype=torch.float32)
        gm = make_fx(fn, tracing_mode="fake")(example)

        with patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}):
            stablehlo, _, _ = convert_fx_to_stablehlo(gm, [example])
        return stablehlo.operation.get_asm()

    def test_unlegalized_group_falls_back_to_world(self):
        """A raw group name has no brackets, so lowering invents the world group."""
        asm = self._broadcast_stablehlo("1", src=2, world_size=4)
        assert "stablehlo.collective_broadcast" in asm
        assert "replica_groups = dense<[[0, 1, 2, 3]]>" in asm

    def test_legalized_subgroup_lowers_to_subgroup(self):
        """The legalized replica_groups string is what reaches the HLO."""
        asm = self._broadcast_stablehlo("[[2, 3]]", src=2, world_size=4)
        assert "stablehlo.collective_broadcast" in asm
        assert "replica_groups = dense<[[2, 3]]>" in asm
