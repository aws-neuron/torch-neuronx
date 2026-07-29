"""Tests for aten::equal — whole-tensor equality returning a host Python bool.

torch.equal(self, other) -> bool. CPU reference semantics (verified on
torch 2.12.1):
- shape mismatch          -> False (no elementwise work)
- dtype mismatch          -> values are promoted and compared; True if all
                             values match (dtype itself is NOT compared)
- empty tensors           -> True when shapes match, False otherwise
- NaN                     -> False (== semantics; equal(t, t) with NaN is False)
- +0.0 vs -0.0            -> True (== semantics)
- cross-device            -> RuntimeError
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

ALL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
]


def _make_pair(dtype, shape=(4, 5)):
    """Build an identical CPU tensor pair of the given dtype."""
    torch.manual_seed(0)
    if dtype == torch.bool:
        a = torch.randint(0, 2, shape, dtype=torch.int8).bool()
    elif dtype.is_floating_point:
        a = (torch.randn(shape) * 10).to(dtype)
    else:
        a = torch.randint(0, 100, shape, dtype=dtype)
    return a, a.detach().clone()


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestEqual:
    """Test suite for aten::equal on the neuron device."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    # ── dispatch identity ────────────────────────────────────────────────

    def test_dispatch_identity(self, device):
        """torch.equal on neuron tensors must dispatch to the Neuron kernel,
        not the CPU fallback, and must return a Python bool."""
        a = torch.randn(8, 8)
        an, bn = a.to(device), a.detach().clone().to(device)
        with track_neuron_ops():
            result = torch.equal(an, bn)
            assert_op_runs_on_neuron("aten::equal")
        assert result is True
        assert isinstance(result, bool)
        assert not isinstance(result, torch.Tensor)

    def test_method_variant(self, device):
        """Tensor.equal method must route through the same kernel."""
        a = torch.randn(3, 3)
        an, bn = a.to(device), a.detach().clone().to(device)
        with track_neuron_ops():
            assert an.equal(bn) is True
            assert_op_runs_on_neuron("aten::equal")

    # ── basic equality / inequality ──────────────────────────────────────

    @pytest.mark.parametrize("shape", [(1,), (7,), (4, 5), (2, 3, 4), (2, 3, 4, 5)])
    def test_equal_identical(self, device, shape):
        a = torch.randn(shape)
        b = a.detach().clone()
        assert torch.equal(a.to(device), b.to(device)) == torch.equal(a, b)
        assert torch.equal(a.to(device), b.to(device)) is True

    @pytest.mark.parametrize("shape", [(1,), (7,), (4, 5), (2, 3, 4)])
    def test_single_element_diff(self, device, shape):
        """A single differing element anywhere must flip the result to False."""
        a = torch.randn(shape)
        b = a.detach().clone()
        flat = b.view(-1)
        flat[flat.numel() // 2] += 1.0
        assert torch.equal(a, b) is False  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is False

    def test_self_comparison(self, device):
        t = torch.randn(5, 5).to(device)
        assert torch.equal(t, t) is True

    def test_scalar_0dim(self, device):
        a = torch.tensor(3.5)
        b = torch.tensor(3.5)
        c = torch.tensor(-3.5)
        assert torch.equal(a.to(device), b.to(device)) is True
        assert torch.equal(a.to(device), c.to(device)) is False

    def test_large_tensor(self, device):
        """Reduction over a large tensor; single diff in the last element."""
        a = torch.randn(512, 513)
        b = a.detach().clone()
        assert torch.equal(a.to(device), b.to(device)) is True
        b[-1, -1] += 1.0
        assert torch.equal(a.to(device), b.to(device)) is False

    # ── shape mismatch: host-side False, no crash ────────────────────────

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [((2, 3), (3, 2)), ((4,), (4, 1)), ((2, 3), (2, 4)), ((5,), (6,)), ((1,), ())],
    )
    def test_shape_mismatch(self, device, shape_a, shape_b):
        a = torch.ones(shape_a)
        b = torch.ones(shape_b)
        assert torch.equal(a, b) is False  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is False

    # ── dtype mismatch: CPU promotes and compares values ─────────────────

    def test_dtype_mismatch_same_values(self, device):
        """CPU torch.equal promotes mixed dtypes and compares values
        (verified: equal(float32 ones, int64 ones) is True on CPU)."""
        a = torch.ones(4, dtype=torch.float32)
        b = torch.ones(4, dtype=torch.int64)
        cpu_result = torch.equal(a, b)
        assert cpu_result is True  # pin the CPU semantics this test asserts
        assert torch.equal(a.to(device), b.to(device)) == cpu_result

    def test_dtype_mismatch_different_values(self, device):
        a = torch.tensor([1.5, 2.0])
        b = torch.tensor([1, 2], dtype=torch.int32)
        assert torch.equal(a, b) is False  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is False

    def test_dtype_mismatch_f16_vs_f32(self, device):
        a = torch.ones(6, dtype=torch.float16)
        b = torch.ones(6, dtype=torch.float32)
        cpu_result = torch.equal(a, b)
        assert torch.equal(a.to(device), b.to(device)) == cpu_result

    # ── empty tensors ────────────────────────────────────────────────────

    def test_empty_tensors(self, device):
        a = torch.tensor([])
        b = torch.tensor([])
        assert torch.equal(a, b) is True  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is True

    def test_empty_same_shape(self, device):
        a = torch.ones(0, 3)
        b = torch.ones(0, 3)
        assert torch.equal(a.to(device), b.to(device)) is True

    def test_empty_different_shape(self, device):
        """(0,3) vs (3,0): both empty but shapes differ -> False (CPU-verified)."""
        a = torch.ones(0, 3)
        b = torch.ones(3, 0)
        assert torch.equal(a, b) is False  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is False

    # ── special float values ─────────────────────────────────────────────

    def test_nan_not_equal(self, device):
        """NaN != NaN under == semantics; torch.equal([nan],[nan]) is False."""
        a = torch.tensor([float("nan")])
        b = torch.tensor([float("nan")])
        assert torch.equal(a, b) is False  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is False

    def test_nan_self_comparison(self, device):
        """CPU: equal(t, t) with a NaN is False even for the same tensor."""
        t = torch.tensor([1.0, float("nan"), 3.0])
        assert torch.equal(t, t) is False  # CPU reference
        tn = t.to(device)
        assert torch.equal(tn, tn) is False

    def test_signed_zero(self, device):
        """+0.0 == -0.0 under IEEE ==, so equal is True."""
        a = torch.tensor([0.0, -0.0])
        b = torch.tensor([-0.0, 0.0])
        assert torch.equal(a, b) is True  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is True

    def test_inf_values(self, device):
        a = torch.tensor([float("inf"), float("-inf"), 1.0])
        b = a.detach().clone()
        assert torch.equal(a.to(device), b.to(device)) is True
        c = torch.tensor([float("-inf"), float("-inf"), 1.0])
        assert torch.equal(a.to(device), c.to(device)) is False

    # ── non-contiguous ───────────────────────────────────────────────────

    def test_noncontiguous_transpose(self, device):
        a = torch.randn(4, 6)
        an = a.to(device).t()
        bn = a.t().contiguous().to(device)
        assert not an.is_contiguous()
        cpu_result = torch.equal(a.t(), a.t().contiguous())
        assert torch.equal(an, bn) == cpu_result
        assert torch.equal(an, bn) is True

    def test_noncontiguous_slice(self, device):
        a = torch.randn(8, 8)
        an = a.to(device)[::2, 1::2]
        bn = a[::2, 1::2].contiguous().to(device)
        assert torch.equal(an, bn) is True
        cn = bn.detach().clone()
        cn[0, 0] += 1.0
        assert torch.equal(an, cn) is False

    def test_noncontiguous_both_sides(self, device):
        a = torch.randn(5, 7)
        an = a.to(device).t()
        bn = a.detach().clone().to(device).t()
        assert not an.is_contiguous() and not bn.is_contiguous()
        assert torch.equal(an, bn) is True

    # ── dtype sweep ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
    def test_all_dtypes_equal(self, device, dtype):
        a, b = _make_pair(dtype)
        cpu_result = torch.equal(a, b)
        assert cpu_result is True
        with track_neuron_ops():
            result = torch.equal(a.to(device), b.to(device))
            assert_op_runs_on_neuron("aten::equal")
        assert result is True

    @pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
    def test_all_dtypes_unequal(self, device, dtype):
        a, b = _make_pair(dtype)
        if dtype == torch.bool:
            b[0, 0] = ~b[0, 0]
        else:
            b[0, 0] = b[0, 0] + 1
        assert torch.equal(a, b) is False  # CPU reference
        assert torch.equal(a.to(device), b.to(device)) is False

    # ── error semantics ──────────────────────────────────────────────────

    def test_cross_device_raises(self, device):
        """CPU/CUDA equal raise on cross-device comparison; neuron must match."""
        a = torch.randn(3)
        b = a.detach().clone().to(device)
        with pytest.raises(RuntimeError, match="[Dd]evice"):
            torch.equal(b, a)
