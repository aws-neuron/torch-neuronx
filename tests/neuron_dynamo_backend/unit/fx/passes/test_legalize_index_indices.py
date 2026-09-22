"""
Unit tests for legalize_index_indices FX pass.

Tests the FX pass that legalizes indices in indexing operations (aten.index.Tensor
and aten.index_put.default) by handling empty index slices and normalizing negative
indices to positive equivalents.
"""

import pytest
import torch
import torch.fx as fx

from torch_neuronx.neuron_dynamo_backend.fx.passes.legalize_index_indices import (
    LegalizeIndexIndices,
)
from torch_neuronx.neuron_dynamo_backend.utils.fx_utils import _add_metadata_to_node

# ============================================================================
# Helpers
# ============================================================================


def _count_ops(gm: fx.GraphModule, target) -> int:
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target == target)


def _build_index_put_graph(
    input_shape: tuple,
    index_shapes: list[tuple | None],
    values_shape: tuple,
    *,
    index_dtype: torch.dtype = torch.int64,
    input_dtype: torch.dtype = torch.float32,
) -> fx.GraphModule:
    """Build an FX graph with a single aten.index_put.default node."""
    graph = fx.Graph()
    inp = graph.placeholder("input")
    _add_metadata_to_node(inp, shape=input_shape, dtype=input_dtype)

    idx_nodes = []
    for i, idx_shape in enumerate(index_shapes):
        if idx_shape is None:
            idx_nodes.append(None)
        else:
            idx = graph.placeholder(f"idx_{i}")
            _add_metadata_to_node(idx, shape=idx_shape, dtype=index_dtype)
            idx_nodes.append(idx)

    values = graph.placeholder("values")
    _add_metadata_to_node(values, shape=values_shape, dtype=input_dtype)

    index_put = graph.call_function(torch.ops.aten.index_put.default, args=(inp, idx_nodes, values))
    _add_metadata_to_node(index_put, shape=input_shape, dtype=input_dtype)
    graph.output(index_put)
    return fx.GraphModule(torch.nn.Module(), graph)


def _build_index_graph(
    input_shape: tuple,
    index_shapes: list[tuple | None],
    output_shape: tuple,
    *,
    index_dtype: torch.dtype = torch.int64,
    input_dtype: torch.dtype = torch.float32,
) -> fx.GraphModule:
    """Build an FX graph with a single aten.index.Tensor node."""
    graph = fx.Graph()
    inp = graph.placeholder("input")
    _add_metadata_to_node(inp, shape=input_shape, dtype=input_dtype)

    idx_nodes = []
    for i, idx_shape in enumerate(index_shapes):
        if idx_shape is None:
            idx_nodes.append(None)
        else:
            idx = graph.placeholder(f"idx_{i}")
            _add_metadata_to_node(idx, shape=idx_shape, dtype=index_dtype)
            idx_nodes.append(idx)

    index_node = graph.call_function(torch.ops.aten.index.Tensor, args=(inp, idx_nodes))
    _add_metadata_to_node(index_node, shape=output_shape, dtype=input_dtype)
    graph.output(index_node)
    return fx.GraphModule(torch.nn.Module(), graph)


# ============================================================================
# Tests
# ============================================================================


class TestNegativeIndexNormalization:
    """Tests for negative index normalization (idx + dim_size)."""

    @pytest.fixture
    def pass_instance(self):
        return LegalizeIndexIndices()

    @pytest.mark.parametrize(
        "op_builder,extra_args",
        [
            pytest.param(
                lambda shapes, idx_shapes: _build_index_put_graph(
                    shapes, idx_shapes, values_shape=(3,)
                ),
                {},
                id="index_put",
            ),
            pytest.param(
                lambda shapes, idx_shapes: _build_index_graph(
                    shapes, idx_shapes, output_shape=(3, 4)
                ),
                {},
                id="index",
            ),
        ],
    )
    def test_single_dim_normalization(self, pass_instance, op_builder, extra_args):
        """Negative indices on a single dimension should be normalized."""
        gm = op_builder((5, 4), [(3,)])
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 1
        assert _count_ops(gm, torch.ops.aten.lt.Tensor) == 1
        assert _count_ops(gm, torch.ops.aten.add.Tensor) == 1

    def test_index_put_multi_dim_normalization(self, pass_instance):
        """Negative indices on multiple dimensions should each be normalized."""
        gm = _build_index_put_graph(
            input_shape=(5, 4, 3),
            index_shapes=[(2,), (2,)],
            values_shape=(2, 3),
        )
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 2
        assert _count_ops(gm, torch.ops.aten.lt.Tensor) == 2

    def test_index_multi_dim_normalization(self, pass_instance):
        """aten.index.Tensor with multiple index dims should each be normalized."""
        gm = _build_index_graph(
            input_shape=(5, 4, 3),
            index_shapes=[(2,), (2,)],
            output_shape=(2, 3),
        )
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 2

    def test_none_indices_skipped(self, pass_instance):
        """None indices (full slices) should be left unchanged."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[None, (3,)],
            values_shape=(5, 3),
        )
        result = pass_instance.call(gm)

        assert result.modified
        # Only the non-None index (dim 1) should be normalized
        assert _count_ops(gm, torch.ops.aten.where.self) == 1

    def test_non_integer_index_skipped(self, pass_instance):
        """Boolean indices should not be normalized."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(5,)],
            values_shape=(4,),
            index_dtype=torch.bool,
        )
        result = pass_instance.call(gm)

        assert not result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 0

    def test_int32_indices_normalized(self, pass_instance):
        """int32 indices should also be normalized."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(3,)],
            values_shape=(3, 4),
            index_dtype=torch.int32,
        )
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 1


class TestEmptyIndexSlice:
    """Tests for empty index slice handling."""

    @pytest.fixture
    def pass_instance(self):
        return LegalizeIndexIndices()

    def test_index_put_empty_slice_replaced_with_input(self, pass_instance):
        """index_put with empty index should be replaced with its input tensor."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(0,)],
            values_shape=(0, 4),
        )
        assert _count_ops(gm, torch.ops.aten.index_put.default) == 1

        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.index_put.default) == 0
        # Output should now be the input placeholder directly
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        source = output_node.args[0]
        assert source.op == "placeholder"
        # First placeholder is the input tensor
        first_placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
        assert source is first_placeholder

    def test_index_empty_slice_replaced_with_empty(self, pass_instance):
        """index with empty index should be replaced with an empty tensor node."""
        gm = _build_index_graph(
            input_shape=(5, 4),
            index_shapes=[(0,)],
            output_shape=(0, 4),
        )
        assert _count_ops(gm, torch.ops.aten.index.Tensor) == 1

        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.index.Tensor) == 0
        assert _count_ops(gm, torch.ops.aten.empty.memory_format) == 1

    def test_index_empty_slice_preserves_output_shape(self, pass_instance):
        """The empty tensor replacement should have the correct output shape metadata."""
        gm = _build_index_graph(
            input_shape=(5, 4, 3),
            index_shapes=[(0,)],
            output_shape=(0, 4, 3),
        )
        result = pass_instance.call(gm)

        assert result.modified
        empty_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.empty.memory_format
        )
        assert list(empty_node.meta["val"].shape) == [0, 4, 3]

    def test_non_empty_index_not_replaced(self, pass_instance):
        """Non-empty indices should not trigger the empty-slice path."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(3,)],
            values_shape=(3, 4),
        )
        result = pass_instance.call(gm)

        assert result.modified  # modified due to negative index normalization
        assert _count_ops(gm, torch.ops.aten.index_put.default) == 1

    def test_mixed_empty_and_nonempty_indices(self, pass_instance):
        """If any index is empty, the whole op should be treated as empty."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(3,), (0,)],
            values_shape=(0,),
        )
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.index_put.default) == 0


class TestNoModification:
    """Tests where the pass should not modify the graph."""

    @pytest.fixture
    def pass_instance(self):
        return LegalizeIndexIndices()

    def test_no_index_ops(self, pass_instance):
        """Graph without any indexing ops should not be modified."""
        graph = fx.Graph()
        x = graph.placeholder("x")
        _add_metadata_to_node(x, shape=(3, 4), dtype=torch.float32)
        add = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        _add_metadata_to_node(add, shape=(3, 4), dtype=torch.float32)
        graph.output(add)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert not result.modified

    def test_boolean_index_only(self, pass_instance):
        """Graph with only boolean indices should not be modified."""
        gm = _build_index_graph(
            input_shape=(5, 4),
            index_shapes=[(5,)],
            output_shape=(3, 4),
            index_dtype=torch.bool,
        )
        result = pass_instance.call(gm)

        assert not result.modified


class TestEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def pass_instance(self):
        return LegalizeIndexIndices()

    def test_missing_index_metadata_raises(self, pass_instance):
        """Index node without 'val' metadata should raise ValueError."""
        graph = fx.Graph()
        inp = graph.placeholder("input")
        _add_metadata_to_node(inp, shape=(5, 4), dtype=torch.float32)
        idx = graph.placeholder("idx")
        # Intentionally not adding metadata to idx
        idx.meta = {}
        values = graph.placeholder("values")
        _add_metadata_to_node(values, shape=(3, 4), dtype=torch.float32)
        index_put = graph.call_function(torch.ops.aten.index_put.default, args=(inp, [idx], values))
        _add_metadata_to_node(index_put, shape=(5, 4), dtype=torch.float32)
        graph.output(index_put)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with pytest.raises(ValueError, match="missing 'val' in its metadata"):
            pass_instance.call(gm)

    def test_index_empty_slice_missing_node_metadata_raises(self, pass_instance):
        """index.Tensor with empty slice but missing node metadata should raise."""
        graph = fx.Graph()
        inp = graph.placeholder("input")
        _add_metadata_to_node(inp, shape=(5, 4), dtype=torch.float32)
        idx = graph.placeholder("idx")
        _add_metadata_to_node(idx, shape=(0,), dtype=torch.int64)
        index_node = graph.call_function(torch.ops.aten.index.Tensor, args=(inp, [idx]))
        # Intentionally not adding metadata to the index node
        index_node.meta = {}
        graph.output(index_node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        with pytest.raises(ValueError, match="missing 'val' in its metadata"):
            pass_instance.call(gm)

    def test_missing_input_metadata_skips_normalization(self, pass_instance):
        """If input tensor has no metadata, normalization should be skipped."""
        graph = fx.Graph()
        inp = graph.placeholder("input")
        inp.meta = {}  # No metadata
        idx = graph.placeholder("idx")
        _add_metadata_to_node(idx, shape=(3,), dtype=torch.int64)
        values = graph.placeholder("values")
        _add_metadata_to_node(values, shape=(3, 4), dtype=torch.float32)
        index_put = graph.call_function(torch.ops.aten.index_put.default, args=(inp, [idx], values))
        _add_metadata_to_node(index_put, shape=(5, 4), dtype=torch.float32)
        graph.output(index_put)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert not result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 0

    def test_multidimensional_index_tensor(self, pass_instance):
        """Multi-dimensional index tensors should be normalized correctly."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(2, 3)],
            values_shape=(2, 3, 4),
        )
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 1
        # Verify the where node has the correct shape metadata
        where_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.where.self
        )
        assert tuple(where_node.meta["val"].shape) == (2, 3)

    def test_pass_reusability(self, pass_instance):
        """Pass instance should be reusable across multiple graphs."""
        gm1 = _build_index_put_graph(input_shape=(5, 4), index_shapes=[(3,)], values_shape=(3, 4))
        gm2 = _build_index_graph(input_shape=(3, 3), index_shapes=[(2,)], output_shape=(2, 3))

        result1 = pass_instance.call(gm1)
        result2 = pass_instance.call(gm2)

        assert result1.modified
        assert result2.modified


class TestMultipleOps:
    """Tests with multiple indexing ops in the same graph."""

    @pytest.fixture
    def pass_instance(self):
        return LegalizeIndexIndices()

    def test_multiple_index_put_nodes(self, pass_instance):
        """Multiple index_put nodes should all be normalized."""
        graph = fx.Graph()
        inp = graph.placeholder("input")
        _add_metadata_to_node(inp, shape=(5, 4), dtype=torch.float32)
        idx1 = graph.placeholder("idx1")
        _add_metadata_to_node(idx1, shape=(3,), dtype=torch.int64)
        idx2 = graph.placeholder("idx2")
        _add_metadata_to_node(idx2, shape=(2,), dtype=torch.int64)
        values1 = graph.placeholder("values1")
        _add_metadata_to_node(values1, shape=(3, 4), dtype=torch.float32)
        values2 = graph.placeholder("values2")
        _add_metadata_to_node(values2, shape=(2, 4), dtype=torch.float32)

        ip1 = graph.call_function(torch.ops.aten.index_put.default, args=(inp, [idx1], values1))
        _add_metadata_to_node(ip1, shape=(5, 4), dtype=torch.float32)
        ip2 = graph.call_function(torch.ops.aten.index_put.default, args=(ip1, [idx2], values2))
        _add_metadata_to_node(ip2, shape=(5, 4), dtype=torch.float32)
        graph.output(ip2)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 2
        assert _count_ops(gm, torch.ops.aten.index_put.default) == 2

    def test_mixed_index_and_index_put(self, pass_instance):
        """Graph with both index and index_put should normalize both."""
        graph = fx.Graph()
        inp = graph.placeholder("input")
        _add_metadata_to_node(inp, shape=(5, 4), dtype=torch.float32)
        idx = graph.placeholder("idx")
        _add_metadata_to_node(idx, shape=(3,), dtype=torch.int64)
        values = graph.placeholder("values")
        _add_metadata_to_node(values, shape=(3, 4), dtype=torch.float32)

        index_node = graph.call_function(torch.ops.aten.index.Tensor, args=(inp, [idx]))
        _add_metadata_to_node(index_node, shape=(3, 4), dtype=torch.float32)
        index_put_node = graph.call_function(
            torch.ops.aten.index_put.default, args=(inp, [idx], values)
        )
        _add_metadata_to_node(index_put_node, shape=(5, 4), dtype=torch.float32)
        graph.output((index_node, index_put_node))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.where.self) == 2

    def test_mixed_empty_and_nonempty_ops(self, pass_instance):
        """One empty-index op and one normal op in the same graph."""
        graph = fx.Graph()
        inp = graph.placeholder("input")
        _add_metadata_to_node(inp, shape=(5, 4), dtype=torch.float32)
        empty_idx = graph.placeholder("empty_idx")
        _add_metadata_to_node(empty_idx, shape=(0,), dtype=torch.int64)
        normal_idx = graph.placeholder("normal_idx")
        _add_metadata_to_node(normal_idx, shape=(3,), dtype=torch.int64)
        values = graph.placeholder("values")
        _add_metadata_to_node(values, shape=(3, 4), dtype=torch.float32)

        # This one should be removed (empty index)
        ip1 = graph.call_function(torch.ops.aten.index_put.default, args=(inp, [empty_idx], values))
        _add_metadata_to_node(ip1, shape=(5, 4), dtype=torch.float32)
        # This one should be normalized
        ip2 = graph.call_function(
            torch.ops.aten.index_put.default, args=(inp, [normal_idx], values)
        )
        _add_metadata_to_node(ip2, shape=(5, 4), dtype=torch.float32)
        graph.output((ip1, ip2))
        gm = fx.GraphModule(torch.nn.Module(), graph)

        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.index_put.default) == 1
        assert _count_ops(gm, torch.ops.aten.where.self) == 1


class TestNormalizationCorrectness:
    """Tests verifying the structure of the normalization subgraph."""

    @pytest.fixture
    def pass_instance(self):
        return LegalizeIndexIndices()

    def test_normalization_subgraph_structure(self, pass_instance):
        """Verify the inserted normalization nodes: scalar(0), lt, scalar(dim_size), add, where."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(3,)],
            values_shape=(3, 4),
        )
        result = pass_instance.call(gm)

        assert result.modified
        assert _count_ops(gm, torch.ops.aten.scalar_tensor.default) == 2  # 0 and dim_size
        assert _count_ops(gm, torch.ops.aten.lt.Tensor) == 1
        assert _count_ops(gm, torch.ops.aten.add.Tensor) == 1
        assert _count_ops(gm, torch.ops.aten.where.self) == 1

    def test_normalization_metadata_types(self, pass_instance):
        """Verify metadata dtypes on inserted normalization nodes."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(3,)],
            values_shape=(3, 4),
            index_dtype=torch.int32,
        )
        pass_instance.call(gm)

        lt_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.lt.Tensor
        )
        assert lt_node.meta["val"].dtype == torch.bool

        where_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.where.self
        )
        assert where_node.meta["val"].dtype == torch.int32

    def test_scalar_tensor_values(self, pass_instance):
        """Verify the scalar tensor args: 0 for comparison, dim_size for offset."""
        gm = _build_index_put_graph(
            input_shape=(7, 4),
            index_shapes=[(3,)],
            values_shape=(3, 4),
        )
        pass_instance.call(gm)

        scalar_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.scalar_tensor.default
        ]
        assert len(scalar_nodes) == 2
        assert scalar_nodes[0].args[0] == 0  # zero for comparison
        assert scalar_nodes[1].args[0] == 7  # dim_size for offset

    def test_scalar_tensor_dtype_matches_index(self, pass_instance):
        """Regression: scalar_tensor must be built with dtype=idx_meta.dtype.

        Without an explicit dtype, scalar_tensor defaults to float32, which
        promotes the subsequent add/where to float32 and produces a float
        index tensor — causing aten.index to fail at runtime with
        "tensors used as indices must be long, int, byte or bool tensors".
        """
        for idx_dtype in (torch.int32, torch.int64):
            gm = _build_index_put_graph(
                input_shape=(7, 4),
                index_shapes=[(3,)],
                values_shape=(3, 4),
                index_dtype=idx_dtype,
            )
            pass_instance.call(gm)

            scalar_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "call_function" and n.target == torch.ops.aten.scalar_tensor.default
            ]
            assert len(scalar_nodes) == 2
            for s in scalar_nodes:
                assert s.kwargs.get("dtype") == idx_dtype, (
                    f"scalar_tensor must carry dtype={idx_dtype} kwarg so the "
                    f"downstream where-normalized index stays integer; got kwargs={s.kwargs}"
                )

    def test_scalar_tensor_dtype_matches_index_on_index_op(self, pass_instance):
        """Same regression as above, verified via aten.index.Tensor path."""
        gm = _build_index_graph(
            input_shape=(7, 4),
            index_shapes=[(3,)],
            output_shape=(3, 4),
            index_dtype=torch.int64,
        )
        pass_instance.call(gm)

        scalar_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.scalar_tensor.default
        ]
        assert len(scalar_nodes) == 2
        for s in scalar_nodes:
            assert s.kwargs.get("dtype") == torch.int64

    def test_where_connects_to_index_put(self, pass_instance):
        """The where node should be used as the new index in the index_put args."""
        gm = _build_index_put_graph(
            input_shape=(5, 4),
            index_shapes=[(3,)],
            values_shape=(3, 4),
        )
        pass_instance.call(gm)

        ip_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.index_put.default
        )
        where_node = next(
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.where.self
        )
        assert ip_node.args[1][0] is where_node


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
