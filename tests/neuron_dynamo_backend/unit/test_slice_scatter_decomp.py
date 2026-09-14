"""Unit tests for the slice_scatter contiguous-write decomposition.

A statically contiguous slice write is served by a zero-padded copy of src masked into
the destination; a strided or symbolic write keeps the scatter. CPU-only.
"""

import collections

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torch_neuronx.neuron_dynamo_backend import decompositions
from torch_neuronx.neuron_dynamo_backend.decompositions import slice_scatter_decomposition

# Static writes the decomposition must serve exactly.
# (input_shape, dim, start, end, label)
STATIC_WRITES = [
    ((4, 3072), -1, 0, 2880, "2880 of 3072"),
    ((4, 3072), -1, 2880, None, "the tail after 2880"),
    ((4, 3072), 1, 0, 2880, "the same span via a positive dim"),
    ((8,), 0, 2, 6, "rank 1, middle span"),
    ((8,), 0, 0, 8, "whole dim"),
    ((8,), 0, 0, None, "whole dim via end=None"),
    ((8,), 0, None, 4, "prefix via start=None"),
    ((8,), 0, 3, None, "suffix"),
    ((8,), 0, -4, None, "negative start"),
    ((8,), 0, 0, -2, "negative end"),
    ((8,), 0, 2, 2, "empty range"),
    ((8,), 0, 6, 100, "end past the extent"),
    ((8,), 0, -100, 4, "start before the beginning"),
    ((2, 3, 4), 1, 1, 3, "middle dim of rank 3"),
    ((2, 3, 4), 0, 1, 2, "outer dim of rank 3"),
]

# Writes that are one contiguous span but not the whole dim, so they must pad and select,
# with the (low, high) pad amounts src is padded by along the written dim.
# (input_shape, dim, start, end, pad_amounts, label)
SPAN_LAYOUTS = [
    ((4, 3072), -1, 0, 2880, (0, 192), "suffix only"),
    ((4, 3072), -1, 2880, None, (2880, 0), "prefix only"),
    ((8,), 0, 2, 6, (2, 2), "prefix and suffix"),
    ((2, 3, 4), 1, 1, 3, (1, 0), "middle dim of rank 3"),
]

# Any of these in the graph means the write went back to one index per element.
PER_ELEMENT = ("scatter", "index_copy", "index_put", "index_select", "_unsafe_masked_index")


def _src_for(inp, dim, start, end, step=1):
    """src must have exactly the extent aten expects; aten does not broadcast it."""
    extent = torch.ops.aten.slice.Tensor(inp, dim, start, end, step)
    return torch.full_like(extent, -7.0)


def test_decomposition_is_registered():
    assert (
        decompositions.neuron_decompositions[torch.ops.aten.slice_scatter.default]
        is slice_scatter_decomposition
    )


@pytest.mark.parametrize(
    "shape,dim,start,end,label", STATIC_WRITES, ids=[w[4] for w in STATIC_WRITES]
)
def test_static_write_matches_reference(shape, dim, start, end, label):
    inp = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    src = _src_for(inp, dim, start, end)

    expected = torch.ops.aten.slice_scatter.default(inp, src, dim, start, end, 1)
    got = slice_scatter_decomposition(inp, src, dim, start, end, 1)
    assert got is not NotImplemented, f"{label}: declined a static write"
    assert got.shape == expected.shape, label
    assert torch.equal(got, expected), label


@pytest.mark.parametrize(
    "shape,dim,start,end,label", STATIC_WRITES, ids=[w[4] for w in STATIC_WRITES]
)
def test_static_write_returns_a_fresh_tensor(shape, dim, start, end, label):
    """A caller that mutates an output aliasing src or input would corrupt the source."""
    inp = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    src = _src_for(inp, dim, start, end)

    got = slice_scatter_decomposition(inp, src, dim, start, end, 1)
    assert got.data_ptr() not in (inp.data_ptr(), src.data_ptr()), label


def test_strided_write_is_declined():
    """step != 1 is not one contiguous span, so the scatter has to stay."""
    inp = torch.arange(8, dtype=torch.float32)
    src = _src_for(inp, 0, 0, 6, 2)
    assert slice_scatter_decomposition(inp, src, 0, 0, 6, 2) is NotImplemented


def test_non_contiguous_input_matches_reference():
    """A transposed destination is masked along its logical dim, as aten slices it."""
    inp = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    assert not inp.is_contiguous()

    for dim, start, end in ((0, 1, 5), (1, 0, 3), (1, 2, None), (0, 0, None)):
        src = _src_for(inp, dim, start, end)
        expected = torch.ops.aten.slice_scatter.default(inp, src, dim, start, end, 1)
        got = slice_scatter_decomposition(inp, src, dim, start, end, 1)
        assert got.shape == expected.shape, (dim, start, end)
        assert torch.equal(got, expected), (dim, start, end)


@pytest.mark.parametrize(
    "start,end", [(0, 4), (0, 8), (2, 2)], ids=["pad and select", "whole dim", "empty range"]
)
def test_write_casts_src_to_destination_dtype(start, end):
    """select would type-promote; aten casts src into the destination instead."""
    inp = torch.zeros(8, dtype=torch.bfloat16)
    extent = torch.ops.aten.slice.Tensor(inp, 0, start, end, 1)
    src = torch.full(extent.shape, 3.0, dtype=torch.float32)

    expected = torch.ops.aten.slice_scatter.default(inp, src.to(torch.bfloat16), 0, start, end, 1)
    got = slice_scatter_decomposition(inp, src, 0, start, end, 1)
    assert got.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(got, expected)


def test_out_of_range_dim_is_declined():
    """A dim outside the input's rank cannot be indexed, so decline it."""
    inp = torch.zeros(4, 4)
    src = torch.ones(4, 4)
    assert slice_scatter_decomposition(inp, src, 2, 0, 4, 1) is NotImplemented
    assert slice_scatter_decomposition(inp, src, -3, 0, 4, 1) is NotImplemented


def _trace_write(shape, dim, start, end, step=1):
    """Trace via the registered op, so a registration regression shows up too."""
    inp = torch.zeros(shape, dtype=torch.float32)
    src = torch.ones_like(torch.ops.aten.slice.Tensor(inp, dim, start, end, step))

    def fn(x, s):
        return torch.ops.aten.slice_scatter.default(x, s, dim, start, end, step)

    gm = make_fx(fn, decomposition_table=decompositions.neuron_decompositions)(inp, src)
    counts = collections.Counter()
    for node in gm.graph.nodes:
        if node.op == "call_function":
            counts[str(node.target)] += 1
    return gm, counts


@pytest.mark.parametrize(
    "shape,dim,start,end,_pad_amounts,label", SPAN_LAYOUTS, ids=[w[5] for w in SPAN_LAYOUTS]
)
def test_static_write_has_no_per_element_form(shape, dim, start, end, _pad_amounts, label):
    _, counts = _trace_write(shape, dim, start, end)
    bad = {k: v for k, v in counts.items() if any(t in k for t in PER_ELEMENT)}
    assert not bad, f"{label}: per-element write form in the graph: {bad}"
    assert counts["aten.constant_pad_nd.default"] == 1, f"{label}: expected one pad"
    assert counts["aten.where.self"] == 1, f"{label}: expected one select"
    assert "aten.cat.default" not in counts, f"{label}: concatenated instead of selecting"


@pytest.mark.parametrize(
    "shape,dim,start,end,pad_amounts,label", SPAN_LAYOUTS, ids=[w[5] for w in SPAN_LAYOUTS]
)
def test_pad_carries_the_span_boundary_not_an_operand(shape, dim, start, end, pad_amounts, label):
    """The boundary lives in the pad attribute, not an operand the backend could shard on."""
    gm, _ = _trace_write(shape, dim, start, end)
    d = dim + len(shape) if dim < 0 else dim

    pad = next(n for n in gm.graph.nodes if str(n.target) == "aten.constant_pad_nd.default")
    # aten pads the last dim first, two entries per dim, so dim d sits at rank-1-d.
    amounts = pad.args[1]
    offset = 2 * (len(shape) - 1 - d)
    assert tuple(amounts[offset : offset + 2]) == pad_amounts, f"{label}: pad {amounts}"
    assert all(
        a == 0 for i, a in enumerate(amounts) if not offset <= i < offset + 2
    ), f"{label}: padded a dim other than {d}: {amounts}"
    assert pad.meta["val"].shape[d] == shape[d], f"{label}: pad did not reach the full extent"

    where = next(n for n in gm.graph.nodes if str(n.target) == "aten.where.self")
    mask = where.args[0].meta["val"]
    assert mask.shape[d] == shape[d], f"{label}: mask is {mask.shape}, not full width on {d}"
    assert mask.dtype == torch.bool, f"{label}: mask dtype {mask.dtype}"


def test_whole_dim_write_is_a_bare_copy():
    """A write covering the whole dim replaces input: nothing to mask against, just a copy."""
    _, counts = _trace_write((4, 8), -1, 0, 8)
    assert counts == {"aten.clone.default": 1}, f"whole-dim write should be one clone, got {counts}"


def test_strided_write_keeps_the_scatter_in_the_graph():
    gm, counts = _trace_write((8,), 0, 0, 6, 2)
    assert any(
        any(t in k for t in PER_ELEMENT) for k in counts
    ), f"expected a per-element form for step=2, got {counts}"

    inp = torch.arange(8, dtype=torch.float32)
    src = torch.ones(3, dtype=torch.float32)
    assert torch.equal(gm(inp, src), torch.ops.aten.slice_scatter.default(inp, src, 0, 0, 6, 2))


def test_symbolic_bound_takes_the_scatter_path():
    """Nothing pins Dynamo to static shapes, so a SymInt start must still decline."""

    def fn(x, s):
        return torch.ops.aten.slice_scatter.default(x, s, 0, x.shape[0] // 2, None, 1)

    inp = torch.arange(8, dtype=torch.float32)
    src = torch.ones(4, dtype=torch.float32)
    gm = make_fx(
        fn, decomposition_table=decompositions.neuron_decompositions, tracing_mode="symbolic"
    )(inp, src)

    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert any(any(p in t for p in PER_ELEMENT) for t in targets), targets
    assert torch.equal(gm(inp, src), torch.ops.aten.slice_scatter.default(inp, src, 0, 4, None, 1))


def test_unbacked_symint_is_declined_without_guarding():
    """Only an unbacked symint proves the guard never compares; a backed one guards silently."""
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    env = ShapeEnv()
    u = env.create_unbacked_symint()
    inp = torch.zeros(8, dtype=torch.float32)
    src = torch.ones(4, dtype=torch.float32)

    assert slice_scatter_decomposition(inp, src, 0, u, None, 1) is NotImplemented
    assert slice_scatter_decomposition(inp, src, 0, 0, u, 1) is NotImplemented
    assert slice_scatter_decomposition(inp, src, 0, 0, 4, u) is NotImplemented
