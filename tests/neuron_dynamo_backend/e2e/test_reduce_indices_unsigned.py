"""
E2E tests for max.dim/min.dim (and the argmax/argmin that decompose to them) on
unsigned integer dtypes.

The reduce region that carries the winning index compares candidate values, and
compare_type selects the comparator rather than annotating it. Emitting SIGNED
for unsigned operands is rejected by the HLO verifier, so these ops used to fail
to compile on ui8/ui32.
"""

import pytest
import torch

# (uint dtype, values whose high bit is set, id)
UNSIGNED_CASES = [
    pytest.param(torch.uint8, [[100, 200, 3, 4], [0, 5, 130, 255]], id="ui8"),
    pytest.param(torch.uint32, [[100, 3000000000, 3, 4], [7, 5, 4294967295, 99]], id="ui32"),
]

SIGNED_CASES = [
    pytest.param(torch.int32, [[100, 20, 3, 4], [-7, 5, 130, 99]], id="i32"),
    pytest.param(torch.int64, [[100, 20, 3, 4], [-7, 5, 130, 99]], id="i64"),
]


def _compile(fn):
    return torch.compile(fn, backend="neuron", fullgraph=True)


def _check(fn, data, dtype):
    """Compare against a reference computed in int64, which is wide enough to
    hold every value in `data` with its true (unsigned) magnitude."""
    ref_in = torch.tensor(data, dtype=torch.int64)
    x = ref_in.to(dtype)

    with torch.no_grad():
        expected = fn(ref_in)
        got = _compile(fn)(x.to("neuron"))

    got = got.cpu().to(torch.int64)
    torch.testing.assert_close(got, expected.to(torch.int64))


class TestMaxDimIndicesUnsigned:
    """max.dim/min.dim keeping the indices result -- the path that emits the
    compare and so the one that carries the compare_type."""

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_max_dim_indices(self, dtype, data):
        _check(lambda a: torch.max(a, dim=-1).indices, data, dtype)

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_max_dim_values(self, dtype, data):
        _check(lambda a: torch.max(a, dim=-1).values, data, dtype)

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_min_dim_indices(self, dtype, data):
        _check(lambda a: torch.min(a, dim=-1).indices, data, dtype)

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_argmax(self, dtype, data):
        _check(lambda a: torch.argmax(a, dim=-1), data, dtype)

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_argmin(self, dtype, data):
        _check(lambda a: torch.argmin(a, dim=-1), data, dtype)


class TestMaxDimIndicesSigned:
    """Controls: signed dtypes must keep signed ordering, including negatives."""

    @pytest.mark.parametrize("dtype,data", SIGNED_CASES)
    def test_argmax(self, dtype, data):
        _check(lambda a: torch.argmax(a, dim=-1), data, dtype)

    @pytest.mark.parametrize("dtype,data", SIGNED_CASES)
    def test_argmin(self, dtype, data):
        _check(lambda a: torch.argmin(a, dim=-1), data, dtype)


class TestAmaxAminUnsigned:
    """amax/amin discard the indices result, so they take the single-region
    reduce path that emits stablehlo.maximum and no compare at all. Included to
    keep that branch covered for unsigned inputs."""

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_amax(self, dtype, data):
        _check(lambda a: torch.amax(a, dim=-1), data, dtype)

    @pytest.mark.parametrize("dtype,data", UNSIGNED_CASES)
    def test_amin(self, dtype, data):
        _check(lambda a: torch.amin(a, dim=-1), data, dtype)
