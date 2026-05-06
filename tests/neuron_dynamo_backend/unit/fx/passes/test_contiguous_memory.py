"""Test for EnsureContiguousInputs FX pass."""

import pytest
import torch

from torch_neuronx.neuron_dynamo_backend.fx.passes.contiguous_memory import (
    EnsureContiguousInputs,
)


class TestEnsureContiguousInputs:
    """Unit tests for EnsureContiguousInputs FX pass."""

    def test_chunk_gets_contiguous_input(self):
        """Test that chunk operations get contiguous inputs after the pass."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                x = x.transpose(1, 2)
                return torch.chunk(x, 2, dim=1)

        gm = torch.fx.symbolic_trace(TestModule())
        result = EnsureContiguousInputs()(gm)

        has_contiguous = any(
            node.target == torch.ops.aten.contiguous.default
            for node in result.graph_module.graph.nodes
        )
        assert has_contiguous, "Expected contiguous() to be inserted before chunk"

    def test_chunk_already_contiguous(self):
        """Test that contiguous is not inserted when input is already contiguous."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.chunk(x, 2, dim=0)

        gm = torch.fx.symbolic_trace(TestModule())
        result = EnsureContiguousInputs()(gm)

        contiguous_count = sum(
            1
            for node in result.graph_module.graph.nodes
            if node.target == torch.ops.aten.contiguous.default
        )
        assert contiguous_count == 0, "Should not insert contiguous for already contiguous input"

    def test_multiple_chunks(self):
        """Test multiple chunk operations."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                x = x.transpose(0, 1)
                a = torch.chunk(x, 2, dim=0)
                b = torch.chunk(x, 3, dim=1)
                return a, b

        gm = torch.fx.symbolic_trace(TestModule())
        result = EnsureContiguousInputs()(gm)

        contiguous_count = sum(
            1
            for node in result.graph_module.graph.nodes
            if node.target == torch.ops.aten.contiguous.default
        )
        assert contiguous_count >= 1, "Should insert contiguous for non-contiguous chunks"

    def test_chunk_with_computation(self):
        """Test chunk followed by computation."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                x = x.transpose(1, 2)
                chunks = torch.chunk(x, 2, dim=1)
                return chunks[0] + chunks[1]

        gm = torch.fx.symbolic_trace(TestModule())
        result = EnsureContiguousInputs()(gm)

        has_contiguous = any(
            node.target == torch.ops.aten.contiguous.default
            for node in result.graph_module.graph.nodes
        )
        assert has_contiguous, "Expected contiguous() to be inserted before chunk"

    def test_no_chunk_operations(self):
        """Test that pass doesn't modify graphs without chunk operations."""

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x.transpose(0, 1) + 1

        gm = torch.fx.symbolic_trace(TestModule())
        result = EnsureContiguousInputs()(gm)

        contiguous_count = sum(
            1
            for node in result.graph_module.graph.nodes
            if node.target == torch.ops.aten.contiguous.default
        )
        assert contiguous_count == 0, "Should not insert contiguous when no chunk operations exist"
