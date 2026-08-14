"""Unit tests for the as_strided / as_strided_scatter contiguous fast paths.

A contiguous access is served by narrow + reshape on the read side and a concat of the
untouched spans on the write side; only overlapping and permuted layouts gather. CPU-only.
"""

import collections
import math

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx

from torch_neuronx.neuron_dynamo_backend import decompositions
from torch_neuronx.neuron_dynamo_backend.decompositions import (
    as_strided,
    as_strided_scatter_decomp,
)

# Contiguous reinterpretations -- the fast path must take these.
# (size, stride, storage_offset, label)
CONTIGUOUS = [
    ((7, 2, 64, 64, 2), (16384, 8192, 128, 2, 1), 0, "KV cache rank5"),
    ((7, 2, 128, 64), (16384, 8192, 64, 1), 16384, "KV cache rank4 + offset"),
    ((3, 4), (4, 1), 0, "plain 2D"),
    ((2, 3, 4), (12, 4, 1), 0, "plain 3D"),
    ((6,), (1,), 0, "rank 1"),
    ((5, 3), (3, 1), 17, "2D at an offset"),
]

# Views that are not a plain row-major reinterpretation, so the write side gathers.
STRIDED = [
    ((3, 4), (1, 3), 0, "transpose"),
    ((4, 8), (16, 2), 0, "step 2 inner"),
    ((5,), (6,), 0, "diagonal of 5x5"),
    ((2, 2), (5, 1), 0, "gappy"),
    ((3, 4), (0, 1), 0, "broadcast dim0"),
]

STORAGE = 1 << 21


@pytest.fixture(scope="module")
def flat():
    return torch.arange(STORAGE, dtype=torch.float32)


@pytest.mark.parametrize("size,stride,offset,label", CONTIGUOUS, ids=[c[3] for c in CONTIGUOUS])
def test_contiguous_view_matches_reference(flat, size, stride, offset, label):
    """Fast path equals torch.as_strided elementwise."""
    expected = torch.as_strided(flat, size, stride, offset)
    got = as_strided(flat, list(size), list(stride), offset)
    assert got.shape == expected.shape, f"{label}: {got.shape} != {expected.shape}"
    assert torch.equal(got, expected), label


@pytest.mark.parametrize("size,stride,offset,label", CONTIGUOUS, ids=[c[3] for c in CONTIGUOUS])
def test_contiguous_view_is_recognized(size, stride, offset, label):
    """The gate drives the write-side concat, so it must accept a contiguous span."""
    assert decompositions._is_static_contiguous_access(size, stride, offset), label


@pytest.mark.parametrize("size,stride,offset,label", STRIDED, ids=[c[3] for c in STRIDED])
def test_strided_views_stay_correct(flat, size, stride, offset, label):
    """Views the contiguous-span gate rejects must still read exactly what aten reads."""
    assert not decompositions._is_static_contiguous_access(
        size, stride, offset
    ), f"{label}: expected non-contiguous"
    expected = torch.as_strided(flat, size, stride, offset)
    got = as_strided(flat, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


@pytest.mark.parametrize("size,stride,offset,label", CONTIGUOUS, ids=[c[3] for c in CONTIGUOUS])
def test_scatter_contiguous_matches_reference(size, stride, offset, label):
    """Write-side fast path equals torch.as_strided_scatter."""
    n = math.prod(size)
    dst = torch.zeros(offset + n + 7, dtype=torch.float32)
    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)

    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert got.shape == expected.shape, label
    assert torch.equal(got, expected), label


@pytest.mark.parametrize("size,stride,offset,label", STRIDED, ids=[c[3] for c in STRIDED])
def test_scatter_strided_falls_back_and_stays_correct(size, stride, offset, label):
    """Non-contiguous destinations keep index_copy semantics."""
    if 0 in stride:
        # A stride-0 view writes several elements to one address; aten rejects
        # it outright, so there is no reference to compare against.
        pytest.skip("aten refuses an overlapping destination")
    n = math.prod(size)
    need = offset + max(1, max(stride)) * max(size) + 1
    dst = torch.zeros(need, dtype=torch.float32)
    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)
    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


# Any of these in the graph means the access went back to one index per element.
PER_ELEMENT = ("scatter", "index_copy", "index_put", "arange", "index_select")

# Contiguous writes, one per multi-span layout, plus the suffix-only layout at KV scale.
# (size, stride, offset, storage, label)
WRITE_LAYOUTS = [
    ((2, 4), (4, 1), 0, 16, "suffix only"),
    ((2, 4), (4, 1), 8, 16, "prefix only"),
    ((2, 4), (4, 1), 4, 16, "prefix and suffix"),
    ((7, 2, 64, 64, 2), (16384, 8192, 128, 2, 1), 0, 2 * 7 * 16384, "KV half"),
]

# Concat operands each layout must produce: src plus the spans it does not touch.
EXPECTED_OPERANDS = {
    "suffix only": 2,
    "prefix only": 2,
    "prefix and suffix": 3,
    "KV half": 2,
}


def _trace_write(size, stride, offset, storage):
    """Trace via the registered op, so a registration regression shows up too."""
    buf = torch.zeros(storage, dtype=torch.float32)
    src = torch.ones(size, dtype=torch.float32)
    _size, _stride, _off = list(size), list(stride), offset

    def fn(x, s):
        return torch.ops.aten.as_strided_scatter.default(x, s, _size, _stride, _off)

    gm = make_fx(fn, decomposition_table=decompositions.neuron_decompositions)(buf, src)
    counts = collections.Counter()
    for node in gm.graph.nodes:
        if node.op == "call_function":
            counts[str(node.target)] += 1
    return gm, counts


@pytest.mark.parametrize(
    "size,stride,offset,storage,label",
    WRITE_LAYOUTS,
    ids=[w[4] for w in WRITE_LAYOUTS],
)
def test_contiguous_write_has_no_per_element_form(size, stride, offset, storage, label):
    """A contiguous write must not emit a scatter or an index tensor."""
    _, counts = _trace_write(size, stride, offset, storage)
    bad = {k: v for k, v in counts.items() if any(t in k for t in PER_ELEMENT)}
    assert not bad, f"{label}: per-element write form in the graph: {bad}"


@pytest.mark.parametrize(
    "size,stride,offset,storage,label",
    WRITE_LAYOUTS,
    ids=[w[4] for w in WRITE_LAYOUTS],
)
def test_contiguous_write_concatenates_untouched_spans(size, stride, offset, storage, label):
    """The write is one concat of src plus only the spans it does not touch."""
    gm, counts = _trace_write(size, stride, offset, storage)
    assert counts["aten.cat.default"] == 1, f"{label}: expected exactly one concat"

    cat = next(n for n in gm.graph.nodes if str(n.target) == "aten.cat.default")
    shapes = [tuple(a.meta["val"].shape) for a in cat.args[0]]
    assert len(shapes) == EXPECTED_OPERANDS[label], f"{label}: concat operands {shapes}"
    assert all(0 not in s for s in shapes), f"{label}: zero-size concat operand {shapes}"
    assert sum(s[0] for s in shapes) == storage, f"{label}: spans {shapes} != {storage}"


def test_full_buffer_write_skips_the_concat():
    """A write covering the whole buffer is a reshape, with nothing to rejoin."""
    _, counts = _trace_write((4, 4), (4, 1), 0, 16)
    assert (
        "aten.cat.default" not in counts
    ), f"nothing to rejoin, but the graph concatenates: {counts}"
    bad = {k: v for k, v in counts.items() if any(t in k for t in PER_ELEMENT)}
    assert not bad, f"per-element write form in the graph: {bad}"


@pytest.mark.parametrize(
    "size,stride,offset,storage,label",
    WRITE_LAYOUTS,
    ids=[w[4] for w in WRITE_LAYOUTS],
)
def test_contiguous_write_matches_reference_at_every_layout(size, stride, offset, storage, label):
    """Every span layout stays numerically equal to torch.as_strided_scatter."""
    torch.manual_seed(0)
    dst = torch.randn(storage, dtype=torch.float32)
    n = math.prod(size)
    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)

    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


# The span layouts above plus the full-buffer write, which needs no concat.
# (size, stride, offset, storage, label)
DTYPE_LAYOUTS = [
    ((2, 4), (4, 1), 0, 16, "suffix only"),
    ((2, 4), (4, 1), 8, 16, "prefix only"),
    ((2, 4), (4, 1), 4, 16, "prefix and suffix"),
    ((4, 4), (4, 1), 0, 16, "full buffer"),
]


@pytest.mark.parametrize(
    "size,stride,offset,storage,label", DTYPE_LAYOUTS, ids=[d[4] for d in DTYPE_LAYOUTS]
)
def test_write_casts_src_to_destination_dtype(size, stride, offset, storage, label):
    """cat would type-promote; aten casts src into the destination instead."""
    dst = torch.zeros(storage, dtype=torch.float32)
    n = math.prod(size)
    src = torch.arange(1, n + 1, dtype=torch.float64).reshape(size)

    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert got.dtype == expected.dtype == torch.float32, label
    assert torch.equal(got, expected), label


# Destinations whose storage order differs from their logical order.
# (factory, label)
NON_CONTIGUOUS_DESTS = [
    (lambda: torch.arange(16, dtype=torch.float32).reshape(4, 4).t(), "transpose 2D"),
    (lambda: torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).permute(2, 0, 1), "permute 3D"),
    (
        lambda: torch.arange(24, dtype=torch.float32).reshape(4, 6).t()[:, :3],
        "transpose then slice",
    ),
]


@pytest.mark.parametrize(
    "factory,label", NON_CONTIGUOUS_DESTS, ids=[d[1] for d in NON_CONTIGUOUS_DESTS]
)
def test_non_contiguous_destination_writes_in_storage_order(factory, label):
    """A dense but non-contiguous destination must match aten, not logical order."""
    dst = factory()
    numel = dst.numel()
    for size, stride, offset in (((2, 4), (4, 1), 4), ((numel,), (1,), 0), ((3,), (1,), 1)):
        n = math.prod(size)
        src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)
        expected = torch.as_strided_scatter(dst, src, size, stride, offset)
        got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
        assert got.shape == expected.shape, f"{label} {size}"
        assert torch.equal(got, expected), f"{label} {size}: {got} != {expected}"


@pytest.mark.parametrize(
    "factory,label", NON_CONTIGUOUS_DESTS, ids=[d[1] for d in NON_CONTIGUOUS_DESTS]
)
def test_non_contiguous_input_reads_storage_order(factory, label):
    """The read side addresses storage too, so its flatten must match aten's.

    Every row here is dense; the input class with no storage-order view is pinned by
    test_gappy_destination_diverges_from_aten.
    """
    src = factory()
    assert not src.is_contiguous(), label
    for size, stride, offset in (((2, 3), (3, 1), 0), ((4,), (2,), 1)):
        expected = torch.as_strided(src, size, stride, offset)
        got = as_strided(src, list(size), list(stride), offset)
        assert torch.equal(got, expected), f"{label} {size}: {got} != {expected}"


# Rank>1 destinations exercise the reshape back, which a 1-D dst cannot.
# (dst_shape, size, stride, offset, label)
RANK_N_WRITES = [
    ((4, 4), (2, 4), (4, 1), 4, "2D dst, middle rows"),
    ((4, 4), (4, 4), (4, 1), 0, "2D dst, whole buffer"),
    ((2, 3, 4), (2, 4), (4, 1), 8, "3D dst, interior span"),
    ((2, 3, 4), (6,), (1,), 18, "3D dst, rank-1 tail"),
]


@pytest.mark.parametrize(
    "dst_shape,size,stride,offset,label", RANK_N_WRITES, ids=[w[4] for w in RANK_N_WRITES]
)
def test_rank_n_destination_write_matches_reference(dst_shape, size, stride, offset, label):
    """A multi-dimensional destination keeps its shape and its untouched values."""
    torch.manual_seed(0)
    dst = torch.randn(dst_shape, dtype=torch.float32)
    n = math.prod(size)
    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)

    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert got.shape == expected.shape == torch.Size(dst_shape), label
    assert torch.equal(got, expected), label


def test_read_of_whole_buffer_is_a_bare_reshape():
    """A whole-buffer read skips the narrow: no slice op, just a reshape."""
    size, storage = (4, 4), 16
    buf = torch.arange(storage, dtype=torch.float32)

    def fn(x):
        return torch.ops.aten.as_strided.default(x, list(size), [4, 1], 0)

    gm = make_fx(fn, decomposition_table=decompositions.neuron_decompositions)(buf)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert not any("slice" in t or "narrow" in t for t in targets), targets
    assert not any(any(p in t for p in PER_ELEMENT) for t in targets), targets
    assert torch.equal(gm(buf), torch.as_strided(buf, size, (4, 1), 0))


def test_symbolic_size_takes_the_gather_path():
    """A symbolic trace must not reach the int-only fast path.

    Nothing pins Dynamo to static shapes, so SymInts can reach these.
    """

    def read(x):
        return torch.ops.aten.as_strided.default(x, [x.shape[0] // 2, 2], [2, 1], 0)

    def write(x, s):
        return torch.ops.aten.as_strided_scatter.default(x, s, [x.shape[0] // 2, 2], [2, 1], 0)

    table = decompositions.neuron_decompositions
    buf = torch.arange(8, dtype=torch.float32)
    src = torch.ones(4, 2, dtype=torch.float32)

    gm = make_fx(read, decomposition_table=table, tracing_mode="symbolic")(buf)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert any("index_select" in t for t in targets), targets
    assert torch.equal(gm(buf), torch.as_strided(buf, (4, 2), (2, 1), 0))

    gm = make_fx(write, decomposition_table=table, tracing_mode="symbolic")(buf, src)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    # index_copy decomposes further to a scatter under make_fx; either form means
    # the write materialized per-element indices.
    assert any("index_copy" in t or "scatter" in t for t in targets), targets
    assert any("arange" in t for t in targets), targets
    assert torch.equal(gm(buf, src), torch.as_strided_scatter(buf, src, (4, 2), (2, 1), 0))


def test_unbacked_symint_is_declined_without_guarding():
    """The gate must reject a symbolic value without asking for its value.

    Only an unbacked symint can prove that: it raises on any comparison, where a backed
    symint would guard silently and pass either way.
    """
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    env = ShapeEnv()
    u = env.create_unbacked_symint()

    assert not decompositions._is_static_contiguous_access([2, 2], [2, 1], u)
    assert not decompositions._is_static_contiguous_access([u, 2], [2, 1], 0)
    assert not decompositions._is_static_contiguous_access([2, 2], [u, 1], 0)


@pytest.mark.parametrize("size,stride,offset,label", CONTIGUOUS, ids=[c[3] for c in CONTIGUOUS])
def test_default_none_offset_behaves_as_zero(size, stride, offset, label):
    """storage_offset is optional in the schema; None must mean 0, not a crash."""
    n = math.prod(size)
    buf = torch.arange(n + 3, dtype=torch.float32)

    assert torch.equal(
        as_strided(buf, list(size), list(stride), None),
        torch.as_strided(buf, size, stride, 0),
    ), label

    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)
    assert torch.equal(
        as_strided_scatter_decomp(buf, src, list(size), list(stride), None),
        torch.as_strided_scatter(buf, src, size, stride, 0),
    ), label


@pytest.mark.parametrize("size,stride,offset,label", STRIDED, ids=[c[3] for c in STRIDED])
def test_gather_path_accepts_a_none_offset(flat, size, stride, offset, label):
    """The fallback adds storage_offset directly, so None must be normalized."""
    assert torch.equal(
        as_strided(flat, list(size), list(stride), None),
        torch.as_strided(flat, size, stride, 0),
    ), label


def test_gather_path_casts_src_to_destination_dtype():
    """index_copy_ rejects a mismatched src; aten casts it into the destination."""
    dst = torch.zeros(16, dtype=torch.float32)
    src = torch.arange(1, 5, dtype=torch.float64).reshape(2, 2)
    size, stride, offset = (2, 2), (5, 1), 0

    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert got.dtype == expected.dtype == torch.float32
    assert torch.equal(got, expected)


@pytest.mark.xfail(
    strict=True,
    reason="a gappy input's storage is unreachable from an FX graph; "
    "the decomposition sees only its dense span",
)
def test_gappy_destination_diverges_from_aten():
    """Documents the one input class the storage-order flatten cannot serve."""
    dst = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, :3]
    flat, _, inverse = decompositions._flatten_in_storage_order(dst)
    # No storage-order view exists, so the flatten falls back to logical order.
    assert inverse is None and torch.equal(flat, dst.reshape(-1))
    src = torch.tensor([100.0, 200.0, 300.0])

    expected = torch.as_strided_scatter(dst, src, (3,), (1,), 1)
    got = as_strided_scatter_decomp(dst, src, [3], [1], 1)
    assert torch.equal(got, expected)


def test_symbolic_storage_offset_takes_the_gather_path():
    """A symbolic storage_offset must not be tested for truthiness.

    Size and stride are concrete here, so only the offset can trip the gate.
    """

    def read(x):
        return torch.ops.aten.as_strided.default(x, [2, 2], [2, 1], x.shape[0] // 2)

    def write(x, s):
        return torch.ops.aten.as_strided_scatter.default(x, s, [2, 2], [2, 1], x.shape[0] // 2)

    table = decompositions.neuron_decompositions
    buf = torch.arange(8, dtype=torch.float32)
    src = torch.ones(2, 2, dtype=torch.float32)

    gm = make_fx(read, decomposition_table=table, tracing_mode="symbolic")(buf)
    assert torch.equal(gm(buf), torch.as_strided(buf, (2, 2), (2, 1), 4))

    gm = make_fx(write, decomposition_table=table, tracing_mode="symbolic")(buf, src)
    assert torch.equal(gm(buf, src), torch.as_strided_scatter(buf, src, (2, 2), (2, 1), 4))


def test_kv_shape_traces_without_index_materialization():
    """The KV-cache view must not put a per-element form in the compiled graph."""
    counts = collections.Counter()

    def fw(gm, inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function":
                counts[str(node.target)] += 1
        return gm.forward

    size, stride = (7, 2, 64, 64, 2), (16384, 8192, 128, 2, 1)
    buf = torch.zeros(7 * 16384, dtype=torch.float32)

    def fn(x):
        return torch.ops.aten.as_strided.default(x, list(size), list(stride), 0) + 1.0

    backend = aot_autograd(fw_compiler=fw, decompositions=decompositions.neuron_decompositions)
    torch.compile(fn, backend=backend, fullgraph=True)(buf)

    assert counts, "the fx compiler never ran"
    bad = {k: v for k, v in counts.items() if any(p in k for p in PER_ELEMENT)}
    assert not bad, f"index materialization leaked into the graph: {bad}"


# Contiguous views whose size-1 dim carries a non-canonical, inert stride.
# (size, stride, offset, label)
SIZE1_INERT_STRIDE = [
    ((3, 4, 1), (4, 1, 4), 0, "permuted size-1 dim"),
    ((3, 1), (1, 3), 0, "transposed (1,3)"),
    ((3, 1, 4), (4, 999, 1), 0, "arbitrary inert stride"),
    ((1, 1), (5, 7), 0, "every dim size 1"),
]


@pytest.mark.parametrize(
    "size,stride,offset,label", SIZE1_INERT_STRIDE, ids=[c[3] for c in SIZE1_INERT_STRIDE]
)
def test_size_1_dim_stride_does_not_block_the_fast_path(flat, size, stride, offset, label):
    """A size-1 dim's only index is 0, so any stride on it denotes the same layout."""
    assert decompositions._is_static_contiguous_access(size, stride, offset), label
    expected = torch.as_strided(flat, size, stride, offset)
    got = as_strided(flat, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


@pytest.mark.parametrize(
    "size,stride,offset,label", SIZE1_INERT_STRIDE, ids=[c[3] for c in SIZE1_INERT_STRIDE]
)
def test_size_1_dim_stride_does_not_block_the_write(size, stride, offset, label):
    """The gate's only caller is the write side, so the exemption has to hold there too."""
    n = math.prod(size)
    dst = torch.zeros(offset + n + 5, dtype=torch.float32)
    src = torch.arange(1, n + 1, dtype=torch.float32).reshape(size)

    expected = torch.as_strided_scatter(dst, src, size, stride, offset)
    got = as_strided_scatter_decomp(dst, src, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


def test_inert_size_1_strides_are_what_torch_calls_contiguous():
    """Ordinary permutes leave an inert stride behind, and torch still reports contiguous."""
    for t in (torch.randn(3, 1, 4).permute(0, 2, 1), torch.randn(1, 3).t()):
        assert t.is_contiguous()
        assert decompositions._is_static_contiguous_access(list(t.shape), list(t.stride()), 0)


# Accesses the plan must decline on its guards, each still correct via the gather.
# (size, stride, offset, label)
PLAN_GUARDS = [
    ((2, 2), (3, 1), 59, "block runs past the buffer"),
    ((4, 8), (16, 1), 8, "whole block past the buffer"),
    ((0, 3), (3, 1), 0, "leading zero-size dim"),
    ((3, 0), (1, 1), 0, "trailing zero-size dim"),
]


@pytest.mark.parametrize("size,stride,offset,label", PLAN_GUARDS, ids=[c[3] for c in PLAN_GUARDS])
def test_plan_guards_decline_and_the_gather_stays_correct(size, stride, offset, label):
    """Declining must fall through to the gather, not raise or read out of bounds."""
    buf = torch.arange(64, dtype=torch.float32)
    assert decompositions._strided_runs_plan(size, stride, offset, buf.numel()) is None, label

    expected = torch.as_strided(buf, size, stride, offset)
    got = as_strided(buf, list(size), list(stride), offset)
    assert got.shape == expected.shape, label
    assert torch.equal(got, expected), label


# Layouts that are not one contiguous span but are still equally spaced contiguous runs.
# (size, stride, offset, label)
RUN_SLICING = [
    ((4, 8), (16, 1), 0, "4 runs of 8"),
    ((4, 8), (16, 2), 0, "step 2 inner"),
    ((5,), (6,), 0, "diagonal of 5x5"),
    ((2, 2), (5, 1), 0, "gappy"),
]

# Layouts that are not expressible as a slice at all.
NOT_RUN_SLICEABLE = [
    ((3, 4), (1, 3), 0, "transpose needs a permute"),
    ((3, 4), (0, 1), 0, "stride 0 overlaps"),
]


@pytest.mark.parametrize("size,stride,offset,label", RUN_SLICING, ids=[c[3] for c in RUN_SLICING])
def test_run_slicing_serves_multi_run_layouts(flat, size, stride, offset, label):
    """Equally spaced runs are a reshape plus a slice, not a gather."""
    assert not decompositions._is_static_contiguous_access(size, stride, offset), label
    assert decompositions._strided_runs_plan(size, stride, offset, flat.numel()) is not None, label
    expected = torch.as_strided(flat, size, stride, offset)
    got = as_strided(flat, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


@pytest.mark.parametrize(
    "size,stride,offset,label", NOT_RUN_SLICEABLE, ids=[c[3] for c in NOT_RUN_SLICEABLE]
)
def test_run_slicing_declines_what_it_cannot_serve(flat, size, stride, offset, label):
    """Overlapping and permuted layouts must still reach the gather."""
    assert decompositions._strided_runs_plan(size, stride, offset, flat.numel()) is None, label
    expected = torch.as_strided(flat, size, stride, offset)
    got = as_strided(flat, list(size), list(stride), offset)
    assert torch.equal(got, expected), label


@pytest.mark.parametrize("size,stride,offset,label", CONTIGUOUS, ids=[c[3] for c in CONTIGUOUS])
def test_one_contiguous_span_is_a_degenerate_run_plan(flat, size, stride, offset, label):
    """The single-span case is the same plan with a full-length run, not a separate path."""
    plan = decompositions._strided_runs_plan(size, stride, offset, flat.numel())
    assert plan is not None, label
    shape, run = plan
    assert run == shape[-1], f"{label}: expected a full-length run, got {run} of {shape[-1]}"


def test_run_slicing_emits_no_per_element_form():
    """A multi-run read traces to slices instead of an index tensor."""
    buf = torch.arange(64, dtype=torch.float32)

    def fn(x):
        return torch.ops.aten.as_strided.default(x, [4, 8], [16, 1], 0)

    gm = make_fx(fn, decomposition_table=decompositions.neuron_decompositions)(buf)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    bad = [t for t in targets if any(p in t for p in PER_ELEMENT)]
    assert not bad, f"run slicing still materialized indices: {bad}"
    assert torch.equal(gm(buf), torch.as_strided(buf, (4, 8), (16, 1), 0))
