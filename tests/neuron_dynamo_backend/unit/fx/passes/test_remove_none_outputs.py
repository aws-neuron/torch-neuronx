"""
Unit tests for remove_none_outputs FX pass.

Tests the FX pass that removes None outputs from backward graphs and preserves
metadata for restoration during execution.
"""

import pytest
import torch
import torch.fx as fx
from torch.fx.node import Node

from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import (
    NoneOutputInfo,
    RemoveNoneOutputs,
)


class TestRemoveNoneOutputs:
    """Test class for None output removal FX pass."""

    def test_basic_none_removal(self):
        """Test basic None output removal functionality."""

        # Create graph with None outputs: (tensor, None, tensor, None)
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x, None, y, None

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata matches actual implementation
        metadata = pass_instance.result
        assert isinstance(metadata, NoneOutputInfo)
        assert metadata.original_output_count == 4
        assert metadata.non_none_positions == [0, 2]  # positions of tensor outputs
        assert metadata.new_output_count == 2

        # Verify graph modification
        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 2  # Only non-None outputs remain

    def test_mixed_outputs_removal(self):
        """Test None removal with mixed tensor/None pattern."""

        # Create graph with pattern: (tensor, None, tensor, None, tensor)
        class TestModule(torch.nn.Module):
            def forward(self, x, y, z):
                return x * 2, None, y + 1, None, z - 1

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata
        metadata = pass_instance.result
        assert metadata.original_output_count == 5
        assert metadata.non_none_positions == [0, 2, 4]  # positions of tensor outputs
        assert metadata.new_output_count == 3

        # Verify graph modification
        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 3  # Three tensor outputs remain

    def test_no_none_outputs(self):
        """Test behavior when graph has no None outputs."""

        # Create graph with only tensor outputs
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x * 2, y + 1

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata
        metadata = pass_instance.result
        assert metadata.original_output_count == 2
        assert metadata.non_none_positions == [0, 1]  # all positions are non-None
        assert metadata.new_output_count == 2

        # Verify graph is unchanged
        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 2

    def test_all_none_outputs(self):
        """Test behavior when all outputs are None."""

        # Create graph with all None outputs
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return None, None, None

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata
        metadata = pass_instance.result
        assert metadata.original_output_count == 3
        assert metadata.non_none_positions == []  # no non-None positions
        assert metadata.new_output_count == 0

        # Verify graph modification
        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 0  # All outputs removed

    def test_single_none_output(self):
        """Test behavior with single None output."""

        # Create graph with single None output
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return None

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata
        metadata = pass_instance.result
        assert metadata.original_output_count == 1
        assert metadata.non_none_positions == []
        assert metadata.new_output_count == 0

        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 0

    def test_single_tensor_output(self):
        """Test behavior with single tensor output."""

        # Create graph with single tensor output
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata
        metadata = pass_instance.result
        assert metadata.original_output_count == 1
        assert metadata.non_none_positions == [0]
        assert metadata.new_output_count == 1

        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 1

    def test_pass_instance_reusability(self):
        """Test that PassBase instance can be reused for multiple graphs."""

        # First graph: (tensor, None)
        class TestModule1(torch.nn.Module):
            def forward(self, x):
                return x, None

        # Second graph: (None, tensor, None)
        class TestModule2(torch.nn.Module):
            def forward(self, x):
                return None, x, None

        gm1 = fx.symbolic_trace(TestModule1())
        gm2 = fx.symbolic_trace(TestModule2())

        pass_instance = RemoveNoneOutputs()

        # Apply to first graph
        modified_gm1 = pass_instance.call(gm1).graph_module
        metadata1 = pass_instance.result
        assert metadata1.original_output_count == 2
        assert metadata1.non_none_positions == [0]
        assert metadata1.new_output_count == 1

        # Apply to second graph (should update metadata)
        modified_gm2 = pass_instance.call(gm2).graph_module
        metadata2 = pass_instance.result
        assert metadata2.original_output_count == 3
        assert metadata2.non_none_positions == [1]
        assert metadata2.new_output_count == 1

        # Verify both graphs were modified correctly
        outputs1 = modified_gm1.graph.output_node().args[0]
        outputs2 = modified_gm2.graph.output_node().args[0]
        assert len(outputs1) == 1
        assert len(outputs2) == 1

    def test_end_to_end_fx_pass(self):
        """Test complete FX pass pipeline with None removal."""

        # Create original graph: (x, None, y*2, None, x+y)
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x, None, y * 2, None, x + y

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module
        metadata = pass_instance.result

        # Test execution of modified graph
        example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
        compressed_outputs = modified_gm(*example_inputs)

        # Verify compressed outputs
        assert len(compressed_outputs) == 3  # Only non-None outputs
        assert torch.equal(compressed_outputs[0], example_inputs[0])  # x passthrough
        assert torch.equal(compressed_outputs[1], example_inputs[1] * 2)  # y*2
        assert torch.equal(compressed_outputs[2], example_inputs[0] + example_inputs[1])  # x+y

        # Verify metadata for restoration
        assert metadata.non_none_positions == [0, 2, 4]
        assert metadata.original_output_count == 5
        assert metadata.new_output_count == 3

    def test_complex_none_pattern(self):
        """Test complex None pattern: (None, tensor, None, None, tensor, None)."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return None, x * 2, None, None, y + 1, None

        gm = fx.symbolic_trace(TestModule())

        # Apply the pass
        pass_instance = RemoveNoneOutputs()
        modified_gm = pass_instance.call(gm).graph_module

        # Verify metadata
        metadata = pass_instance.result
        assert metadata.original_output_count == 6
        assert metadata.non_none_positions == [1, 4]  # positions of tensor outputs
        assert metadata.new_output_count == 2

        # Verify graph modification
        output_node = modified_gm.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 2  # Only two tensor outputs remain
