"""
Unit tests for random_op_legalization FX pass.

Tests the FX pass that legalizes random operations (e.g., native_dropout) by
converting them to deterministic operations with random mask inputs.
"""

import operator

import pytest
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from torch_neuronx.neuron_dynamo_backend.fx.passes.random_op_legalization import (
    RandomOpLegalization,
)
from torch_neuronx.neuron_dynamo_backend.utils.stablehlo_utils import (
    NativeDropoutOp,
    RandomInputInfo,
)


class DropoutGraphBuilder:
    """Helper class to construct FX graphs with native_dropout operations."""

    @staticmethod
    def create_single_dropout_graph(shape, dropout_prob: float, train: bool) -> fx.GraphModule:
        """Create a graph with a single native_dropout operation."""

        class SingleDropoutModule(torch.nn.Module):
            def __init__(self, p, train_mode):
                super().__init__()
                self.p = p
                self.train_mode = train_mode

            def forward(self, x):
                output, _ = torch.ops.aten.native_dropout.default(x, self.p, self.train_mode)
                return output

        module = SingleDropoutModule(dropout_prob, train)
        gm = torch.fx.symbolic_trace(module)
        example_input = torch.randn(*shape)
        ShapeProp(gm).propagate(example_input)
        return gm

    @staticmethod
    def create_dropout_with_mask_output(shape, dropout_prob: float, train: bool) -> fx.GraphModule:
        """Create a graph where the dropout mask is used as output."""

        class DropoutWithMaskModule(torch.nn.Module):
            def __init__(self, p, train_mode):
                super().__init__()
                self.p = p
                self.train_mode = train_mode

            def forward(self, x):
                _, mask = torch.ops.aten.native_dropout.default(x, self.p, self.train_mode)
                return mask

        module = DropoutWithMaskModule(dropout_prob, train)
        gm = torch.fx.symbolic_trace(module)
        example_input = torch.randn(*shape)
        ShapeProp(gm).propagate(example_input)
        return gm

    @staticmethod
    def create_multi_dropout_graph(shapes, probs, trains) -> fx.GraphModule:
        """Create a graph with multiple native_dropout operations."""
        assert len(shapes) == len(probs) == len(trains)

        class MultiDropoutModule(torch.nn.Module):
            def __init__(self, probs, trains):
                super().__init__()
                assert len(probs) == 3 and len(trains) == 3
                self.probs = probs
                self.trains = trains

            def forward(self, input):
                x, _ = torch.ops.aten.native_dropout.default(
                    input[0], self.probs[0], self.trains[0]
                )
                a = torch.nn.functional.softmax(x, dim=0)
                y, _ = torch.ops.aten.native_dropout.default(a, self.probs[1], self.trains[1])
                b = torch.nn.functional.softplus(y, beta=1, threshold=20)
                z, _ = torch.ops.aten.native_dropout.default(b, self.probs[2], self.trains[2])
                return torch.add(x, torch.add(y, z))

        module = MultiDropoutModule(probs, trains)
        gm = torch.fx.symbolic_trace(module)
        example_inputs = tuple(torch.randn(*shape) for shape in shapes)
        ShapeProp(gm).propagate(*example_inputs)
        return gm

    @staticmethod
    def create_no_dropout_graph(shape) -> fx.GraphModule:
        """Create a graph without any dropout operations."""

        class NoDropoutModule(torch.nn.Module):
            def forward(self, x):
                return x * 2 + 1

        module = NoDropoutModule()
        gm = torch.fx.symbolic_trace(module)
        example_input = torch.randn(*shape)
        ShapeProp(gm).propagate(example_input)
        return gm


def count_placeholders(gm: fx.GraphModule) -> int:
    """Count placeholder (input) nodes in the graph."""
    return sum(1 for node in gm.graph.nodes if node.op == "placeholder")


def count_ops(gm: fx.GraphModule, target) -> int:
    """Count occurrences of a specific operation in the graph."""
    return sum(1 for node in gm.graph.nodes if node.op == "call_function" and node.target == target)


# Test parameter sets
TEST_SHAPES = [
    pytest.param((4,), id="1d"),
    pytest.param((2, 3), id="2d"),
    pytest.param((2, 3, 4), id="3d"),
    pytest.param((1, 2, 3, 4), id="4d"),
]

TEST_PROBABILITIES = [
    pytest.param(0.0, id="p=0"),
    pytest.param(0.1, id="p=0.1"),
    pytest.param(0.5, id="p=0.5"),
    pytest.param(0.9, id="p=0.9"),
    pytest.param(1.0, id="p=1"),
]

TEST_TRAIN_MODES = [
    pytest.param(True, id="train=True"),
    pytest.param(False, id="train=False"),
]


class TestRandomOpLegalization:
    """Test class for random op legalization FX pass."""

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    @pytest.mark.parametrize("prob", TEST_PROBABILITIES)
    @pytest.mark.parametrize("train", TEST_TRAIN_MODES)
    def test_dropout_legalization(self, shape, prob: float, train: bool):
        """Test dropout legalization with various shapes, probabilities, and train modes.

        Expected behavior:
        - train=False or p=0: identity (ones mask, no new input, no random op)
        - train=True and p=1: zeros output (zeros mask, no new input, no random op)
        - train=True and 0 < p < 1: random mask added (new input, mul ops, random op recorded)
        """
        gm = DropoutGraphBuilder.create_single_dropout_graph(
            shape=shape, dropout_prob=prob, train=train
        )
        assert count_ops(gm, torch.ops.aten.native_dropout.default) == 1
        original_input_count = count_placeholders(gm)

        pass_instance = RandomOpLegalization()
        result = pass_instance.call(gm)

        # validate result metadata
        modified_gm = result.graph_module
        metadata = pass_instance.result
        new_input_count = count_placeholders(modified_gm)
        assert result.modified
        assert count_ops(modified_gm, torch.ops.aten.native_dropout.default) == 0
        assert metadata.original_input_count == original_input_count
        assert metadata.new_input_count == new_input_count

        # validate FX graph properties
        needs_random_input = train and 0 < prob < 1
        is_identity = (not train) or prob == 0
        is_zeros = train and prob == 1
        if needs_random_input:
            # Random mask case: new input added, mul ops present, op recorded
            assert new_input_count == original_input_count + 1
            assert count_ops(modified_gm, torch.ops.aten.mul.Tensor) == 2
            assert len(metadata.ops) == 1
            op = metadata.ops[0]
            assert isinstance(op, NativeDropoutOp)
            assert op.probability == prob
            assert op.train is train
            assert op.shape == shape
        elif is_identity:
            # Identity case: no new input, ones mask, no op recorded
            assert new_input_count == original_input_count
            assert count_ops(modified_gm, torch.ops.aten.ones.default) >= 1
            assert len(metadata.ops) == 0
        elif is_zeros:
            # Zeros case: no new input, zeros mask and output, no op recorded
            assert new_input_count == original_input_count
            assert count_ops(modified_gm, torch.ops.aten.zeros.default) >= 2
            assert len(metadata.ops) == 0

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_mask_output_replacement(self, shape):
        """Test that mask output is correctly replaced with input mask."""
        gm = DropoutGraphBuilder.create_dropout_with_mask_output(
            shape=shape, dropout_prob=0.5, train=True
        )
        pass_instance = RandomOpLegalization()
        result = pass_instance.call(gm)

        modified_gm = result.graph_module
        output_node = modified_gm.graph.output_node()
        assert result.modified
        assert output_node is not None
        assert len(output_node.args) == 1 and output_node.args[0] is not None

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_no_dropout_operations(self, shape):
        """Test behavior when graph has no dropout operations."""
        gm = DropoutGraphBuilder.create_no_dropout_graph(shape=shape)
        original_input_count = count_placeholders(gm)
        pass_instance = RandomOpLegalization()
        result = pass_instance.call(gm)
        metadata = pass_instance.result

        assert not result.modified
        assert metadata.original_input_count == original_input_count
        assert metadata.new_input_count == original_input_count
        assert len(metadata.ops) == 0

    def test_multiple_dropout_operations(self):
        """Test graph with multiple dropout operations."""
        shapes = [(2, 3), (2, 3), (2, 3)]
        probs = [0.3, 0.5, 0.7]
        trains = [True, True, True]
        gm = DropoutGraphBuilder.create_multi_dropout_graph(shapes, probs, trains)
        original_input_count = count_placeholders(gm)
        assert count_ops(gm, torch.ops.aten.native_dropout.default) == 3

        pass_instance = RandomOpLegalization()
        result = pass_instance.call(gm)

        # validate returned metadata matches original order of ops in module
        modified_gm = result.graph_module
        new_input_count = count_placeholders(modified_gm)
        metadata = pass_instance.result
        assert result.modified
        assert count_ops(modified_gm, torch.ops.aten.native_dropout.default) == 0
        assert new_input_count == original_input_count + 3
        assert len(metadata.ops) == 3
        assert metadata.ops[0].probability == 0.3
        assert metadata.ops[1].probability == 0.5
        assert metadata.ops[2].probability == 0.7
        assert metadata.ops[0].input_position == 0
        assert metadata.ops[1].input_position == 1
        assert metadata.ops[2].input_position == 2

    def test_mixed_train_modes(self):
        """Test multiple dropouts with different train modes."""
        shapes = [(2, 3), (2, 3), (2, 3)]
        probs = [0.3, 0.5, 0.7]
        trains = [True, False, True]
        gm = DropoutGraphBuilder.create_multi_dropout_graph(shapes, probs, trains)
        original_input_count = count_placeholders(gm)

        pass_instance = RandomOpLegalization()
        result = pass_instance.call(gm)

        # validate returned metadata only returned for train nodes
        metadata = pass_instance.result
        new_input_count = count_placeholders(result.graph_module)
        assert result.modified
        assert len(metadata.ops) == 2
        assert metadata.ops[0].train
        assert metadata.ops[0].probability == 0.3
        assert metadata.ops[0].input_position == 0
        assert metadata.ops[1].train
        assert metadata.ops[1].probability == 0.7
        assert metadata.ops[1].input_position == 1
        assert new_input_count == original_input_count + 2

    def test_pass_instance_reusability(self):
        """Test that pass instance can be reused for multiple graphs."""
        gm1 = DropoutGraphBuilder.create_single_dropout_graph(
            shape=(2, 3), dropout_prob=0.3, train=True
        )
        gm2 = DropoutGraphBuilder.create_single_dropout_graph(
            shape=(4, 5), dropout_prob=0.7, train=True
        )

        pass_instance = RandomOpLegalization()

        result1 = pass_instance.call(gm1)
        metadata1 = pass_instance.result
        assert metadata1 is not None
        assert result1.modified is True
        assert len(metadata1.ops) == 1
        assert metadata1.ops[0].probability == 0.3
        assert metadata1.ops[0].train is True
        assert metadata1.ops[0].shape == (2, 3)

        result2 = pass_instance.call(gm2)
        metadata2 = pass_instance.result
        assert metadata2 is not None
        assert result2.modified is True
        assert len(metadata2.ops) == 1
        assert metadata2.ops[0].probability == 0.7
        assert metadata2.ops[0].train is True
        assert metadata2.ops[0].shape == (4, 5)

    def test_dropout_probability_must_be_constant(self):
        """Test that non-constant dropout probability raises an error."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        p = graph.placeholder("p")
        x.meta["val"] = torch.randn(2, 3)
        dropout = graph.call_function(torch.ops.aten.native_dropout.default, args=(x, p, True))
        getitem = graph.call_function(operator.getitem, args=(dropout, 0))
        graph.output(getitem)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        pass_instance = RandomOpLegalization()
        with pytest.raises(ValueError, match="must be a constant"):
            pass_instance.call(gm)
