"""Tests for aten::tril_indices / aten::triu_indices — integer-exact factory ops.

torch.tril_indices(row, col, offset) -> (2, N) int tensor of lower-triangle
coordinates; triu_indices mirrors it for the upper triangle. CPU reference
semantics (verified on torch 2.13.0):
- dtype: int32 and int64 only; anything else raises
  RuntimeError('"tril_indices" not implemented for ...')
- offset beyond the matrix in either direction yields the full or empty set
- row == 0 or col == 0 yields shape (2, 0)
- no out= at the Python level, and OpInfo carries supports_out=False, so the
  .out overload is not registered (nothing dispatches it)

The implementations are integer-exact (see python_ops/nki_kernels/tril_indices.py and
neuron_dynamo_backend/decompositions.py), so every comparison here is torch.equal, never
assert_close.
"""

import itertools

import pytest
import torch

from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops
from torch_neuronx.neuron_dynamo_backend import decompositions as decomp_mod
from torch_neuronx.neuron_dynamo_backend.decompositions import (
    tril_indices_decomposition,
    triu_indices_decomposition,
)
from torch_neuronx.python_ops.nki_kernels import packed_interleave as nki_pack
from torch_neuronx.python_ops.nki_kernels import tril_indices as nki_trilu
from torch_neuronx.utils import get_logical_neuron_cores

SHAPES = [
    (0, 0),
    (0, 5),
    (5, 0),
    (1, 1),
    (1, 7),
    (7, 1),
    (3, 4),
    (4, 3),
    (7, 19),
    (19, 7),
    (16, 16),
    (37, 23),
    (64, 64),
    (128, 96),
    (2048, 2048),
]
OFFSETS = [-129, -17, -1, 0, 1, 17, 129]
INT_DTYPES = [torch.int32, torch.int64]
NKI_CORNER_SHAPES = [(1, 1), (1, 7), (7, 1), (23, 37), (37, 23), (129, 129)]
SEQ_LEN_SHAPES = [(2048, 2048), (4096, 4096), (8192, 8192)]

_OPS = {
    "tril_indices": (torch.tril_indices, tril_indices_decomposition, False),
    "triu_indices": (torch.triu_indices, triu_indices_decomposition, True),
}


class _RecordingLauncher:
    """Wrap a kernel launcher and record the grid of every launch."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.grids = []

    def __getitem__(self, grid):
        inner = self._wrapped[grid]

        def launch(*args, **kwargs):
            self.grids.append(tuple(inner.grid))
            return inner(*args, **kwargs)

        return launch


class TestTrilTriuIndicesGeometry:
    """CPU-only checks of the geometry the kernel is given. No hardware."""

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_geometry_matches_aten_sizes(self, op_name):
        """The host geometry must agree with aten on the output length, for every offset.

        The kernel trusts these five numbers, so a geometry error is a silent wrong
        result. aten is the reference.
        """
        aten_fn, _, upper = _OPS[op_name]
        for (row, col), offset in itertools.product(SHAPES, OFFSETS):
            n_trap, n_rect, _, _, _ = nki_trilu._trilu_sizes(row, col, offset, upper)
            expected = aten_fn(row, col, offset).shape[1]
            assert n_trap + n_rect == expected, (op_name, row, col, offset)

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_geometry_can_fail(self, op_name):
        """Negative control: the geometry of a different offset must not match."""
        aten_fn, _, upper = _OPS[op_name]
        n_trap, n_rect, _, _, _ = nki_trilu._trilu_sizes(8, 8, 0, upper)
        assert n_trap + n_rect != aten_fn(8, 8, 1).shape[1]

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_trapezoid_rows_and_edge_are_consistent(self, op_name):
        """n_trap must equal the sum of the row widths the edge and row count imply."""
        _, _, upper = _OPS[op_name]
        for (row, col), offset in itertools.product(SHAPES, OFFSETS):
            n_trap, _, n_trap_rows, edge, _ = nki_trilu._trilu_sizes(row, col, offset, upper)
            if n_trap_rows == 0:
                assert n_trap == 0, (row, col, offset)
                continue
            # Rows grow by one from the edge width, which is how C(r) = r*(r+k)/2 counts.
            assert n_trap == sum(edge + i for i in range(n_trap_rows)), (row, col, offset)

    @pytest.mark.parametrize(
        "row,col,offset,upper,expected",
        [
            (46340, 46340, 0, False, True),
            (46341, 46341, 0, False, False),
            (46341, 2, 0, False, True),
            (2, 1_000_000, 0, False, True),
            # A skewed triu has few elements, so the product bound admits it.
            (2, 1_000_000, 0, True, True),
            (46340, 46340, 0, True, True),
            (65535, 65535, 0, False, False),
            (65535, 65535, 0, True, False),
            # Every offset is covered, at any size the products allow.
            (8192, 8192, 1, False, True),
            (8192, 8192, -1, True, True),
            (46340, 46340, 1, False, True),
            (10, 10, 2, False, True),
            (10, 10, -2, True, True),
            # An offset past the diagonal makes the full matrix, still covered.
            (5, 5, 10, False, True),
            (5, 5, -10, True, True),
            # An empty result has no work, so the decomposition makes it.
            (5, 5, -10, False, False),
            (5, 5, 10, True, False),
            (0, 5, 0, False, False),
        ],
    )
    def test_nki_eligibility_bounds(self, row, col, offset, upper, expected):
        assert nki_trilu.trilu_nki_eligible(row, col, offset, upper) is expected

    def test_fence_product_bound_is_int32(self):
        """The kernel forms r*(r + k) in int32, so the bound is that product."""
        assert nki_trilu.INT32_MAX == 2**31 - 1
        n_trap, _, n_trap_rows, edge, _ = nki_trilu._trilu_sizes(46340, 46340, 0, False)
        assert nki_trilu._trilu_fence_product(n_trap_rows, edge) <= nki_trilu.INT32_MAX
        assert n_trap == 46340 * 46341 // 2


class TestTrilTriuIndicesDevice:
    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_matches_cpu_bit_exact(self, op_name, dtype):
        aten_fn, _, _ = _OPS[op_name]
        with track_neuron_ops():
            got = aten_fn(37, 23, 2, dtype=dtype, device="neuron")
            assert got.device.type == "neuron"
            assert_op_runs_on_neuron(f"aten::{op_name}")
        assert torch.equal(got.cpu(), aten_fn(37, 23, 2, dtype=dtype))

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("offset", OFFSETS)
    def test_offsets(self, op_name, offset):
        aten_fn, _, _ = _OPS[op_name]
        got = aten_fn(19, 7, offset, device="neuron")
        assert torch.equal(got.cpu(), aten_fn(19, 7, offset))

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("row,col", [(0, 0), (0, 5), (5, 0), (1, 1)])
    def test_edge_shapes(self, op_name, row, col):
        aten_fn, _, _ = _OPS[op_name]
        got = aten_fn(row, col, 0, device="neuron")
        expected = aten_fn(row, col, 0)
        assert got.shape == expected.shape
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_boundary_heavy_large_shape(self, op_name):
        """1024x1024 hits every triangular-number boundary, where the float
        inversion is most likely to misround."""
        aten_fn, _, _ = _OPS[op_name]
        got = aten_fn(1024, 1024, 0, device="neuron")
        assert torch.equal(got.cpu(), aten_fn(1024, 1024, 0))

    def test_default_dtype_is_int64(self):
        got = torch.tril_indices(4, 4, device="neuron")
        assert got.dtype == torch.int64

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("row,col", SEQ_LEN_SHAPES)
    def test_sequence_length_shapes_bit_exact(self, op_name, row, col):
        """Attention-sized inputs; 8192x8192 is 33.5M pairs, above the 2^24 exact limit."""
        aten_fn, _, _ = _OPS[op_name]
        with track_neuron_ops():
            got = aten_fn(row, col, 0, device="neuron")
            assert_op_runs_on_neuron(f"aten::{op_name}")
        assert torch.equal(got.cpu(), aten_fn(row, col, 0))

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize("row,col", NKI_CORNER_SHAPES)
    def test_nki_path_corner_shapes_bit_exact(self, op_name, dtype, row, col):
        aten_fn, _, _ = _OPS[op_name]
        with track_neuron_ops():
            got = aten_fn(row, col, 0, dtype=dtype, device="neuron")
            assert_op_runs_on_neuron(f"aten::{op_name}")
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), aten_fn(row, col, 0, dtype=dtype))

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_offset_nonzero_runs_the_kernel(self, op_name):
        aten_fn, _, _ = _OPS[op_name]
        for offset in (3, -3):
            got = aten_fn(64, 64, offset, device="neuron")
            assert torch.equal(got.cpu(), aten_fn(64, 64, offset))

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize(
        "row,col", [(37, 23), (23, 37), (1024, 1024), (4096, 4096), (8192, 8192)]
    )
    def test_compile_matches_cpu_bit_exact(self, op_name, row, col):
        """Every offset-0 shape runs the shared NKI kernel in-graph; results match CPU."""
        aten_fn, _, _ = _OPS[op_name]
        compiled = torch.compile(lambda: aten_fn(row, col, 0, device="neuron"), backend="neuron")
        got = compiled()
        assert got.device.type == "neuron"
        assert torch.equal(got.cpu(), aten_fn(row, col, 0))

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_decomposition_routes_by_size_on_neuron(self, op_name, monkeypatch):
        """Kernel for every non-empty shape and offset; fenced only for an empty result."""
        _, neuron_fn, upper = _OPS[op_name]
        calls = []
        sentinel = object()
        monkeypatch.setattr(decomp_mod, "trilu_nki_indices", lambda *a: calls.append(a) or sentinel)
        # Every size and every offset runs the kernel.
        assert neuron_fn(23, 37, 0, dtype=torch.int64, device="neuron") is sentinel
        assert neuron_fn(8192, 8192, 0, dtype=torch.int64, device="neuron") is sentinel
        assert neuron_fn(23, 37, 3, dtype=torch.int64, device="neuron") is sentinel
        assert neuron_fn(23, 37, -3, dtype=torch.int64, device="neuron") is sentinel
        assert calls == [
            (upper, 23, 37, 0, torch.int64, "neuron"),
            (upper, 8192, 8192, 0, torch.int64, "neuron"),
            (upper, 23, 37, 3, torch.int64, "neuron"),
            (upper, 23, 37, -3, torch.int64, "neuron"),
        ]
        # An empty result has no work, so the fenced decomposition makes it.
        empty_offset = 40 if upper else -40
        got = neuron_fn(23, 37, empty_offset, dtype=torch.int64, device="neuron")
        assert got.cpu().numel() == 0
        assert len(calls) == 4
        # Above the int32 bound the kernel refuses; nothing else is exact either.
        with pytest.raises(NotImplementedError, match="index elements"):
            neuron_fn(65535, 65535, 0, dtype=torch.int64, device="neuron")
        assert len(calls) == 4

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    @pytest.mark.parametrize(
        "row,col,offset", [(23, 37, 0), (4, 3, -4), (20, 21, -20), (4, 3, 5), (37, 23, 3)]
    )
    def test_decomposition_direct_call_matches_cpu(self, op_name, dtype, row, col, offset):
        """The fenced body run op by op on the device, with each half empty in turn."""
        aten_fn, neuron_fn, _ = _OPS[op_name]
        got = neuron_fn(row, col, offset, dtype=dtype, device="neuron")
        assert torch.equal(got.cpu(), aten_fn(row, col, offset, dtype=dtype))

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("offset", [3, -3])
    def test_compile_offset_runs_the_kernel(self, op_name, offset, monkeypatch):
        """An offset runs the same kernel in-graph; the fenced body is not used on neuron."""
        aten_fn, _, upper = _OPS[op_name]
        calls = []
        real = decomp_mod.trilu_nki_indices

        def recording(*a):
            calls.append(a[:4])
            return real(*a)

        monkeypatch.setattr(decomp_mod, "trilu_nki_indices", recording)
        compiled = torch.compile(lambda: aten_fn(37, 23, offset, device="neuron"), backend="neuron")
        got = compiled()
        assert calls == [(upper, 37, 23, offset)]
        assert torch.equal(got.cpu(), aten_fn(37, 23, offset))

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_compile_kernel_route_bit_exact(self, op_name, dtype, monkeypatch):
        """The kernel route under torch.compile, checked plane by plane.

        int64 packs the kernel's int32 output into pairs with the shared interleave kernel,
        then one dtype view; int32 takes the plain output.
        """
        aten_fn, _, upper = _OPS[op_name]
        calls = []
        real = decomp_mod.trilu_nki_indices

        def recording(*a):
            calls.append(a[:4])
            return real(*a)

        monkeypatch.setattr(decomp_mod, "trilu_nki_indices", recording)
        launches = _RecordingLauncher(nki_trilu._wrapped_trilu_nki)
        monkeypatch.setattr(nki_trilu, "_wrapped_trilu_nki", launches)
        interleave = _RecordingLauncher(nki_pack.packed_interleave_hop)
        monkeypatch.setattr(nki_pack, "packed_interleave_hop", interleave)
        compiled = torch.compile(
            lambda: aten_fn(37, 23, 0, dtype=dtype, device="neuron"), backend="neuron"
        )
        got = compiled()
        assert calls == [(upper, 37, 23, 0)]
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), aten_fn(37, 23, 0, dtype=dtype))
        # int64 must reach the interleave kernel. A convert of the kernel output
        # instead costs 73x on trn2 at 8192x8192, and no other test would notice.
        assert launches.grids == [(int(get_logical_neuron_cores()),)]
        assert interleave.grids == (
            [(int(get_logical_neuron_cores()),)] if dtype == torch.int64 else []
        )

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_eager_int64_packs_with_the_shared_kernel(self, op_name, dtype, monkeypatch):
        """The eager body packs an int64 request with the shared interleave kernel.

        The traced body calls the pack, and a cached graph is not re-traced, so the recorder
        would see nothing on a warm cache. Turn the cache off for the call. Meta inference and
        the trace each run the body once, so int64 can record more than one launch; each launch
        must use the target LNC grid. int32 never packs.
        """
        from torch_neuronx.python_ops.compilation import hlo_cache

        aten_fn, _, _ = _OPS[op_name]
        monkeypatch.setattr(hlo_cache, "cache_enabled", lambda *a, **k: False, raising=True)
        interleave = _RecordingLauncher(nki_pack.packed_interleave_hop)
        monkeypatch.setattr(nki_pack, "packed_interleave_hop", interleave)
        got = aten_fn(41, 29, 0, dtype=dtype, device="neuron")
        assert torch.equal(got.cpu(), aten_fn(41, 29, 0, dtype=dtype))
        if dtype == torch.int64:
            assert interleave.grids and set(interleave.grids) == {
                (int(get_logical_neuron_cores()),)
            }
        else:
            assert interleave.grids == []

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_oversize_shape_fails_loudly(self, op_name):
        """Above 2^31 index elements no route is exact, so refuse instead of guessing."""
        aten_fn, _, _ = _OPS[op_name]
        with pytest.raises(NotImplementedError, match="index elements"):
            aten_fn(65535, 65535, 0, device="neuron")

    @pytest.mark.parametrize("op_name", list(_OPS))
    def test_rejects_non_integer_dtype(self, op_name):
        aten_fn, _, _ = _OPS[op_name]
        with pytest.raises(RuntimeError, match=f'"{op_name}" not implemented for'):
            aten_fn(3, 4, 0, dtype=torch.float32, device="neuron")

    @pytest.mark.parametrize("op_name", list(_OPS))
    @pytest.mark.parametrize("road", ["eager", "compile"])
    def test_layout_and_pin_memory_follow_the_dispatcher(self, op_name, road):
        """The dispatcher refuses a sparse layout before this op runs, and it ignores
        pin_memory for every device op; both roads inherit that, and neither checks."""
        aten_fn, _, _ = _OPS[op_name]

        def call(**kw):
            fn = lambda: aten_fn(5, 5, 0, device="neuron", **kw)  # noqa: E731
            return torch.compile(fn, backend="neuron")() if road == "compile" else fn()

        assert torch.equal(call(pin_memory=True).cpu(), aten_fn(5, 5, 0))
        with pytest.raises(NotImplementedError, match=op_name):
            call(layout=torch.sparse_coo)
