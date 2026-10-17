"""
Unit tests for remove_none_outputs FX pass.

Tests the FX pass that removes None outputs from backward graphs and records
IOMutation metadata for restoration during execution.
"""

import pytest
import torch
import torch.fx as fx
from torch.fx.node import Node

from torch_neuronx.neuron_dynamo_backend.fx.io_mutation import MutationKind
from torch_neuronx.neuron_dynamo_backend.fx.passes.neuron_fx_pass import NeuronFXPassState
from torch_neuronx.neuron_dynamo_backend.fx.passes.remove_none_outputs import (
    RemoveNoneOutputs,
)


def _make_state():
    """Create a minimal NeuronFXPassState for testing."""
    from unittest.mock import MagicMock

    return NeuronFXPassState(job=MagicMock(), model_name="test", dumping_enabled=False)


class TestRemoveNoneOutputs:
    """Test class for None output removal FX pass."""

    def test_basic_none_removal(self):
        """Test basic None output removal functionality."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x, None, y, None

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        # Verify IOMutation recorded
        assert len(state.pass_mutations[0].output_mutations) == 1
        mutation = state.pass_mutations[0].output_mutations[0]
        assert mutation.kind == MutationKind.REMOVE
        assert mutation.indices == [1, 3]
        assert mutation.transform_key == "none"
        assert mutation.pass_name == "RemoveNoneOutputs"

        # Verify graph modification
        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 2

    def test_mixed_outputs_removal(self):
        """Test None removal with mixed tensor/None pattern."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y, z):
                return x * 2, None, y + 1, None, z - 1

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        assert len(state.pass_mutations[0].output_mutations) == 1
        assert state.pass_mutations[0].output_mutations[0].indices == [1, 3]

        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 3

    def test_no_none_outputs(self):
        """Test behavior when graph has no None outputs."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x * 2, y + 1

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        # No mutation recorded when there are no Nones
        assert len(state.pass_mutations) == 0
        assert result.modified is False

        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 2

    def test_all_none_outputs(self):
        """Test behavior when all outputs are None."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return None, None, None

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        assert len(state.pass_mutations[0].output_mutations) == 1
        assert state.pass_mutations[0].output_mutations[0].indices == [0, 1, 2]

        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 0

    def test_single_none_output(self):
        """Test behavior with single None output."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return None

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        assert len(state.pass_mutations[0].output_mutations) == 1
        assert state.pass_mutations[0].output_mutations[0].indices == [0]

        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 0

    def test_single_tensor_output(self):
        """Test behavior with single tensor output (no tuple wrapping)."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        assert len(state.pass_mutations) == 0
        assert result.modified is False

    def test_end_to_end_fx_pass(self):
        """Test complete FX pass pipeline with None removal."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x, None, y * 2, None, x + y

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)
        modified_gm = result.graph_module

        # Test execution of modified graph
        example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
        compressed_outputs = modified_gm(*example_inputs)

        assert len(compressed_outputs) == 3
        assert torch.equal(compressed_outputs[0], example_inputs[0])
        assert torch.equal(compressed_outputs[1], example_inputs[1] * 2)
        assert torch.equal(compressed_outputs[2], example_inputs[0] + example_inputs[1])

        assert state.pass_mutations[0].output_mutations[0].indices == [1, 3]

    def test_complex_none_pattern(self):
        """Test complex None pattern: (None, tensor, None, None, tensor, None)."""

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return None, x * 2, None, None, y + 1, None

        gm = fx.symbolic_trace(TestModule())

        state = _make_state()
        pass_instance = RemoveNoneOutputs(state)
        result = pass_instance.call(gm)

        assert len(state.pass_mutations[0].output_mutations) == 1
        assert state.pass_mutations[0].output_mutations[0].indices == [0, 2, 3, 5]

        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 2

    def test_without_state(self):
        """Test that the pass works without state (no mutation recorded)."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x, None

        gm = fx.symbolic_trace(TestModule())

        pass_instance = RemoveNoneOutputs()
        result = pass_instance.call(gm)

        assert result.modified is True
        output_node = result.graph_module.graph.output_node()
        outputs = output_node.args[0]
        assert len(outputs) == 1
