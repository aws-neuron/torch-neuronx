"""Tests for on-device random operations (uniform_, normal_, random_, random_.from)."""

import pytest
import torch

from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops


class TestUniform:
    def test_uniform_default_range(self):
        """Test uniform_ with default [0, 1) range."""
        t = torch.empty(1024, device="neuron")
        with track_neuron_ops():
            t.uniform_()
            assert_op_runs_on_neuron("aten::uniform_")
        vals = t.cpu()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_uniform_custom_range(self):
        """Test uniform_ with custom [from, to) range."""
        t = torch.empty(1024, device="neuron")
        with track_neuron_ops():
            t.uniform_(-5.0, 5.0)
            assert_op_runs_on_neuron("aten::uniform_")
        vals = t.cpu()
        assert vals.min() >= -5.0
        assert vals.max() < 5.0

    def test_uniform_shape_preserved(self):
        """Test that uniform_ preserves tensor shape."""
        t = torch.empty(8, 16, 32, device="neuron")
        t.uniform_()
        assert t.shape == (8, 16, 32)

    def test_uniform_inplace(self):
        """Test that uniform_ modifies the tensor in place."""
        t = torch.zeros(256, device="neuron")
        t.uniform_()
        # Should no longer be all zeros (statistically near-impossible)
        assert not torch.all(t.cpu() == 0.0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_uniform_dtypes(self, dtype):
        """Test uniform_ with different floating point dtypes."""
        t = torch.empty(512, dtype=dtype, device="neuron")
        t.uniform_()
        assert t.dtype == dtype
        vals = t.cpu().float()
        assert vals.min() >= 0.0
        # bfloat16 rounding can push values to exactly 1.0
        assert vals.max() <= 1.0

    def test_uniform_negative_range(self):
        """Test uniform_ with a range entirely below zero."""
        t = torch.empty(1024, device="neuron")
        t.uniform_(-10.0, -2.0)
        vals = t.cpu()
        assert vals.min() >= -10.0
        assert vals.max() < -2.0

    def test_uniform_zero_width_range(self):
        """Test uniform_ with from == to fills with that single value."""
        t = torch.empty(256, device="neuron")
        t.uniform_(3.5, 3.5)
        torch.testing.assert_close(t.cpu(), torch.full((256,), 3.5))

    def test_uniform_spans_range(self):
        """Test uniform_ populates the whole range, not just part of it."""
        t = torch.empty(4096, device="neuron")
        t.uniform_(0.0, 10.0)
        vals = t.cpu()
        # With 4096 samples the mean lands near the midpoint and both halves
        # of the range are populated.
        assert abs(vals.mean().item() - 5.0) < 0.5
        assert (vals < 5.0).any()
        assert (vals >= 5.0).any()

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.complex64,
            pytest.param(
                torch.complex32,
                marks=pytest.mark.xfail(
                    reason="random ops complex32 (ComplexHalf) unsupported: experimental in "
                    "torch and the on-device complex write drops the imaginary part",
                    strict=False,
                ),
            ),
        ],
    )
    def test_uniform_complex_dtypes(self, dtype):
        """uniform_ supports complex on-device: real & imag each ~U[0, 1)."""
        t = torch.empty(10000, dtype=dtype, device="neuron")
        with track_neuron_ops():
            t.uniform_()
            assert_op_runs_on_neuron("aten::uniform_")
        assert t.dtype == dtype
        parts = torch.view_as_real(t.cpu().to(torch.complex64))
        assert torch.isfinite(parts).all()
        assert parts.min().item() >= 0.0 and parts.max().item() < 1.0
        assert abs(parts.mean().item() - 0.5) < 0.05

    def test_uniform_integer_dtype_raises(self):
        """Test uniform_ on an integer tensor raises, matching CPU."""
        t = torch.empty(16, dtype=torch.int32, device="neuron")
        with pytest.raises(NotImplementedError, match="check_uniform_bounds"):
            t.uniform_()

    def test_uniform_inverted_range_raises(self):
        """Test uniform_ rejects from > to, matching CPU."""
        t = torch.empty(16, device="neuron")
        with pytest.raises(RuntimeError, match="uniform_ expects to return"):
            t.uniform_(5.0, 1.0)

    def test_uniform_span_wider_than_dtype_raises(self):
        """Test uniform_ rejects a span the dtype cannot represent, matching CPU."""
        t = torch.empty(16, device="neuron")
        finfo = torch.finfo(torch.float32)
        with pytest.raises(RuntimeError, match="uniform_ expects to-from"):
            t.uniform_(finfo.min, finfo.max)


class TestNormal:
    def test_normal_default_params(self):
        """Test normal_ with default mean=0, std=1."""
        t = torch.empty(10000, device="neuron")
        with track_neuron_ops():
            t.normal_()
            assert_op_runs_on_neuron("aten::normal_")
        vals = t.cpu()
        # With 10k samples, mean should be close to 0 and std close to 1
        assert abs(vals.mean().item()) < 0.1
        assert abs(vals.std().item() - 1.0) < 0.1

    def test_normal_custom_params(self):
        """Test normal_ with custom mean and std."""
        t = torch.empty(10000, device="neuron")
        with track_neuron_ops():
            t.normal_(mean=5.0, std=2.0)
            assert_op_runs_on_neuron("aten::normal_")
        vals = t.cpu()
        assert abs(vals.mean().item() - 5.0) < 0.2
        assert abs(vals.std().item() - 2.0) < 0.2

    def test_normal_shape_preserved(self):
        """Test that normal_ preserves tensor shape."""
        t = torch.empty(4, 8, 16, device="neuron")
        t.normal_()
        assert t.shape == (4, 8, 16)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_normal_dtypes(self, dtype):
        """Test normal_ with different floating point dtypes."""
        t = torch.empty(4096, dtype=dtype, device="neuron")
        t.normal_()
        assert t.dtype == dtype
        vals = t.cpu().float()
        # Basic sanity: values should span a reasonable range
        assert vals.min() < -1.0
        assert vals.max() > 1.0

    def test_normal_zero_std(self):
        """Test normal_ with std=0 fills with the mean."""
        t = torch.empty(256, device="neuron")
        t.normal_(mean=2.0, std=0.0)
        torch.testing.assert_close(t.cpu(), torch.full((256,), 2.0))

    def test_normal_is_symmetric(self):
        """Test normal_ produces a roughly symmetric distribution about the mean."""
        t = torch.empty(10000, device="neuron")
        t.normal_()
        vals = t.cpu()
        below = (vals < 0).sum().item()
        # A symmetric distribution splits near 50/50; allow generous slack so
        # the test is not flaky.
        assert 4000 < below < 6000

    def test_normal_negative_std_raises(self):
        """Test normal_ rejects a negative std, matching CPU."""
        t = torch.empty(16, device="neuron")
        with pytest.raises(RuntimeError, match="normal expects std >= 0.0"):
            t.normal_(mean=0.0, std=-1.0)

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.complex64,
            pytest.param(
                torch.complex32,
                marks=pytest.mark.xfail(
                    reason="random ops complex32 (ComplexHalf) unsupported: experimental in "
                    "torch and the on-device complex write drops the imaginary part",
                    strict=False,
                ),
            ),
        ],
    )
    def test_normal_complex_dtypes(self, dtype):
        """normal_ supports complex on-device: real & imag each ~N(0, 1)."""
        t = torch.empty(20000, dtype=dtype, device="neuron")
        with track_neuron_ops():
            t.normal_()
            assert_op_runs_on_neuron("aten::normal_")
        assert t.dtype == dtype
        parts = torch.view_as_real(t.cpu().to(torch.complex64))
        assert torch.isfinite(parts).all()
        assert abs(parts.mean().item()) < 0.1
        assert abs(parts.std().item() - 1.0) < 0.1

    def test_normal_integer_dtype_raises(self):
        """Test normal_ on an integer tensor raises, matching CPU."""
        t = torch.empty(16, dtype=torch.int32, device="neuron")
        with pytest.raises(NotImplementedError, match="normal_kernel_cpu"):
            t.normal_()


class TestRandom:
    def test_random_int_tensor(self):
        """Test random_ on integer tensor."""
        t = torch.empty(1024, dtype=torch.int32, device="neuron")
        with track_neuron_ops():
            t.random_()
            assert_op_runs_on_neuron("aten::random_")
        vals = t.cpu()
        # Should have diverse values (not all the same)
        assert vals.unique().numel() > 10

    def test_random_bool_tensor(self):
        """Test random_ on boolean tensor."""
        t = torch.empty(1024, dtype=torch.bool, device="neuron")
        t.random_()
        vals = t.cpu()
        # Should have both True and False
        assert vals.any()
        assert not vals.all()

    def test_random_float_tensor(self):
        """Test random_ on float tensor fills with integers."""
        t = torch.empty(1024, dtype=torch.float32, device="neuron")
        with track_neuron_ops():
            t.random_()
            assert_op_runs_on_neuron("aten::random_")
        vals = t.cpu()
        assert vals.min() >= 0.0

    def test_random_shape_preserved(self):
        """Test that random_ preserves tensor shape."""
        t = torch.empty(4, 8, 16, dtype=torch.int32, device="neuron")
        t.random_()
        assert t.shape == (4, 8, 16)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
    def test_random_default_is_non_negative(self, dtype):
        """Test bare random_ never draws negatives.

        ATen's random_ uses from=0, so signed dtypes stay non-negative
        (see test_random_default / test_random_full_range upstream).
        """
        t = torch.empty(2048, dtype=dtype, device="neuron")
        t.random_()
        assert t.cpu().to(torch.double).min().item() >= 0

    def test_random_writes_every_element(self):
        """Test random_ fills the whole tensor, not just part of it."""
        t = torch.zeros(2048, dtype=torch.int32, device="neuron")
        t.random_()
        # Bare random_ draws from a huge range, so a zero left behind means an
        # element was never written.
        assert (t.cpu() != 0).all()

    def test_random_full_int32_range(self):
        """Test random_ over the entire int32 range.

        The span (2**32) does not fit in int32, so this exercises the raw-bits
        path rather than the modulo or float-scaling paths.
        """
        iinfo = torch.iinfo(torch.int32)
        t = torch.empty(4096, dtype=torch.int32, device="neuron")
        t.random_(iinfo.min, None)
        vals = t.cpu().to(torch.double)
        # Values must cover both signs, which the float-scaling fallback
        # cannot do without losing the low bits.
        assert vals.min().item() < 0
        assert vals.max().item() > 0


class TestRandomFrom:
    def test_random_from_range(self):
        """Test random_.from with explicit [from, to) range."""
        t = torch.empty(1024, dtype=torch.int32, device="neuron")
        with track_neuron_ops():
            t.random_(10, 20)
            assert_op_runs_on_neuron("aten::random_.from")
        vals = t.cpu()
        assert vals.min() >= 10
        assert vals.max() < 20

    def test_random_from_shape_preserved(self):
        """Test that random_.from preserves tensor shape."""
        t = torch.empty(8, 16, dtype=torch.int64, device="neuron")
        t.random_(0, 100)
        assert t.shape == (8, 16)

    def test_random_from_narrow_range(self):
        """Test random_.from with a narrow range to verify bounds."""
        t = torch.empty(1024, dtype=torch.int32, device="neuron")
        t.random_(5, 8)
        vals = t.cpu()
        assert vals.min() >= 5
        assert vals.max() <= 7  # exclusive upper bound

    def test_random_from_single_value_range(self):
        """Test random_.from with a range holding exactly one value."""
        t = torch.empty(512, dtype=torch.int32, device="neuron")
        t.random_(7, 8)
        assert (t.cpu() == 7).all()

    def test_random_from_negative_range(self):
        """Test random_.from with a range entirely below zero."""
        t = torch.empty(1024, dtype=torch.int32, device="neuron")
        t.random_(-20, -10)
        vals = t.cpu()
        assert vals.min() >= -20
        assert vals.max() < -10

    def test_random_from_spans_range(self):
        """Test random_.from reaches both ends of a small range."""
        t = torch.empty(4096, dtype=torch.int32, device="neuron")
        t.random_(0, 4)
        vals = t.cpu()
        # Every value in [0, 4) should appear across 4096 draws.
        assert set(vals.unique().tolist()) == {0, 1, 2, 3}

    @pytest.mark.parametrize(("from_", "to", "expected"), [(0, 1, False), (1, 2, True)])
    def test_random_from_bool_honors_bounds(self, from_, to, expected):
        """Test random_.from on bool respects a single-value range.

        random_(0, 1) can only yield False and random_(1, 2) only True — a
        bare coin flip would pass the both-values check in
        test_random_bool_tensor while being wrong here.
        """
        t = torch.empty(512, dtype=torch.bool, device="neuron")
        t.random_(from_, to)
        assert (t.cpu() == expected).all()

    def test_random_from_bool_full_range(self):
        """Test random_.from on bool over [0, 2) yields both values."""
        t = torch.empty(1024, dtype=torch.bool, device="neuron")
        t.random_(0, 2)
        vals = t.cpu()
        assert vals.any()
        assert not vals.all()

    def test_random_from_inverted_range_raises(self):
        """Test random_.from rejects from >= to, matching CPU."""
        t = torch.empty(16, dtype=torch.int32, device="neuron")
        with pytest.raises(RuntimeError, match="random_ expects 'from' to be less than 'to'"):
            t.random_(10, 5)

    def test_random_from_out_of_bounds_raises(self):
        """Test random_.from rejects a 'from' outside the dtype range, matching CPU."""
        t = torch.empty(16, dtype=torch.int32, device="neuron")
        with pytest.raises(RuntimeError, match="from is out of bounds"):
            t.random_(torch.iinfo(torch.int64).min, 10)

    def test_random_from_to_out_of_bounds_raises(self):
        """Test random_.from rejects a 'to' outside the dtype range, matching CPU."""
        t = torch.empty(16, dtype=torch.int32, device="neuron")
        with pytest.raises(RuntimeError, match="to - 1 is out of bounds"):
            t.random_(0, torch.iinfo(torch.int64).max)


class TestReproducibility:
    def test_uniform_reproducible_with_manual_seed(self):
        """Test that manual_seed produces reproducible uniform_ results after warmup."""
        t = torch.empty(128, device="neuron")
        t.uniform_()

        torch.neuron.manual_seed(42)
        t1 = torch.empty(128, device="neuron")
        t1.uniform_()

        torch.neuron.manual_seed(42)
        t2 = torch.empty(128, device="neuron")
        t2.uniform_()

        torch.testing.assert_close(t1.cpu(), t2.cpu())

    def test_normal_reproducible_with_manual_seed(self):
        """Test that manual_seed produces reproducible normal_ results after warmup."""
        t = torch.empty(128, device="neuron")
        t.normal_()

        torch.neuron.manual_seed(42)
        t1 = torch.empty(128, device="neuron")
        t1.normal_()

        torch.neuron.manual_seed(42)
        t2 = torch.empty(128, device="neuron")
        t2.normal_()

        torch.testing.assert_close(t1.cpu(), t2.cpu())

    def test_random_reproducible_with_manual_seed(self):
        """Test that manual_seed produces reproducible random_ results after warmup."""
        t = torch.empty(128, dtype=torch.int32, device="neuron")
        t.random_()

        torch.neuron.manual_seed(7)
        t1 = torch.empty(128, dtype=torch.int32, device="neuron")
        t1.random_()

        torch.neuron.manual_seed(7)
        t2 = torch.empty(128, dtype=torch.int32, device="neuron")
        t2.random_()

        torch.testing.assert_close(t1.cpu(), t2.cpu())

    def test_successive_draws_differ(self):
        """Test that consecutive draws are not identical.

        Guards against a cached graph returning the same values every call.
        """
        t1 = torch.empty(1024, device="neuron")
        t1.uniform_()
        first = t1.cpu().clone()
        t1.uniform_()
        assert not torch.equal(first, t1.cpu())


class TestGeneratorFallback:
    @pytest.mark.skip(reason="Generator fallback relies on C++ dispatcher error recovery")
    def test_uniform_with_generator_falls_back(self):
        """Test that passing a generator falls back to CPU and produces correct results."""
        g = torch.Generator()
        g.manual_seed(42)
        t = torch.empty(1024, device="neuron")
        t.uniform_(generator=g)
        vals = t.cpu()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    @pytest.mark.skip(reason="Generator fallback relies on C++ dispatcher error recovery")
    def test_normal_with_generator_falls_back(self):
        """Test that normal_ with generator falls back to CPU and produces correct results."""
        g = torch.Generator()
        g.manual_seed(123)
        t = torch.empty(10000, device="neuron")
        t.normal_(generator=g)
        vals = t.cpu()
        assert abs(vals.mean().item()) < 0.1
        assert abs(vals.std().item() - 1.0) < 0.1
