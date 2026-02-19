"""
Unit tests for collective_legalization fx pass.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch.fx import GraphModule, Node

from torch_neuronx.neuron_dynamo_backend.fx.passes import collective_legalization
from torch_neuronx.neuron_dynamo_backend.fx.passes.collective_legalization import (
    CollectiveLegalization,
)


class TestGetProcessGroupRanks:
    """Test get_process_group_ranks function"""

    @patch("torch.distributed.distributed_c10d._resolve_process_group")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.is_initialized")
    def test_get_process_group_ranks_fallback_initialized(
        self, mock_is_init, mock_world_size, mock_resolve_pg
    ):
        """Test fallback behavior when process group
        resolution fails but distributed is initialized"""
        # Mock resolution failure
        mock_resolve_pg.side_effect = Exception("Failed to resolve process group")
        mock_is_init.return_value = True
        mock_world_size.return_value = 4

        result = collective_legalization._get_process_group_ranks("invalid_group")

        assert result == [0, 1, 2, 3]  # Should fallback to world group

    @patch("torch.distributed.distributed_c10d._resolve_process_group")
    @patch("torch.distributed.is_initialized")
    def test_get_process_group_ranks_fallback_not_initialized(self, mock_is_init, mock_resolve_pg):
        """Test fallback behavior when distributed is not initialized"""
        # Mock resolution failure and no distributed initialization
        mock_resolve_pg.side_effect = Exception("Failed to resolve process group")
        mock_is_init.return_value = False

        result = collective_legalization._get_process_group_ranks("invalid_group")

        assert result == [0]  # Should return single rank for non-distributed case


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
        mock_get_ranks.return_value = [0, 1, 2, 3]

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
        mock_get_ranks.return_value = [0, 2, 4, 6]

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
        mock_get_ranks.return_value = [1, 3, 5, 7]

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
        mock_get_ranks.return_value = [0, 1]

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
    def test_multiple_collective_ops_legalization(self, mock_get_ranks):
        """Test legalization of multiple collective operations"""
        # Mock different return values for different calls
        mock_get_ranks.side_effect = [[0, 1, 2, 3], [0, 1], [2, 3]]

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
        mock_get_ranks.return_value = [0, 1]

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
                "group_0": [0, 1, 2, 3],
                "group_1": [0, 1],
                "group_2": [2, 3],
            }
            return group_mapping.get(group_name, [0])

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
        mock_get_ranks.return_value = [0, 1]

        # Test each supported op
        supported_ops = [
            ("_c10d_functional.all_reduce.default", 2),
            ("_c10d_functional.all_gather_into_tensor.default", 2),
            ("_c10d_functional.reduce_scatter_tensor.default", 3),
            ("_c10d_functional.all_to_all_single.default", 3),
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
