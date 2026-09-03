"""Phase 1 acceptance tests: the error model, verified on CPU with no device.

TestAccumulatorDiscriminator validates that the measured relative error of a bf16 accumulator
differs from an fp32 one by orders of magnitude -- which is why the flat budget can drop the
``u_acc`` term without losing the failure mode it predicted.
"""

from __future__ import annotations

import math

import pytest
import torch

from . import budgets
from .budgets import ARITHMETIC_BUDGET_UNITS, error_budget
from .conftest import _cell
from .formats import BF16, FP32, get_format
from .metrics import (
    FROBENIUS_RATIO_THRESHOLD,
    FrobeniusVerdict,
    cell_id,
    frobenius_ratio,
    norms,
    rel_err_units,
    seed_for,
    seeded,
    to_units,
)

U_BF16 = 2.0**-8
U_FP32 = 2.0**-24


# ── 1. a known injected error produces the expected unit count ───────────────


class TestMetricArithmetic:
    def test_injected_error_reads_back_exactly(self):
        ref = torch.full((100,), 2.0, dtype=torch.float64)
        actual = ref.clone()
        actual[7] += 3.0 * U_BF16 * 2.0  # 3 units at the reference's scale (max|R| = 2.0)

        n = norms(actual, ref)
        assert n["max_abs_ref"] == pytest.approx(2.0)
        assert to_units(n["max_abs_err"], n["max_abs_ref"], U_BF16) == pytest.approx(3.0)

    def test_exact_match_is_zero_error(self):
        ref = torch.randn(64, dtype=torch.float64)
        assert rel_err_units(norms(ref.clone(), ref), U_BF16) == {"max_rel_err_units": 0.0}

    def test_zero_reference_does_not_silently_pass(self):
        """An all-zero reference has no scale. Exact -> 0 units; anything else -> inf."""
        zero = torch.zeros(8, dtype=torch.float64)
        assert to_units(0.0, 0.0, U_BF16) == 0.0
        assert to_units(1e-30, 0.0, U_BF16) == math.inf

        wrong = zero.clone()
        wrong[0] = 1e-30
        assert rel_err_units(norms(wrong, zero), U_BF16)["max_rel_err_units"] == math.inf

    def test_one_bad_element_is_not_diluted_by_tensor_size(self):
        """L-inf reports the worst element, whatever the tensor size around it."""
        ref = torch.full((4096,), 1.0, dtype=torch.float64)

        spike = ref.clone()
        spike[0] += 100.0 * U_BF16
        assert rel_err_units(norms(spike, ref), U_BF16)["max_rel_err_units"] == pytest.approx(100.0)

    def test_nan_and_inf_outputs_are_distinguishable_from_the_metric_alone(self):
        inf = math.inf

        def measure(actual, ref):
            kw = {"dtype": torch.float64}
            n = norms(torch.tensor(actual, **kw), torch.tensor(ref, **kw))
            return to_units(n["max_abs_err"], n["max_abs_ref"], U_BF16)

        assert measure([inf, 1.0], [3.0, 1.0]) == inf  # output overflowed: wrong, countable
        assert math.isnan(measure([3.0, 1.0], [inf, 1.0]))  # reference overflowed: not comparable
        assert math.isnan(measure([inf, 1.0], [inf, 1.0]))  # both: inf - inf, not comparable
        assert math.isnan(measure([math.nan, 1.0], [3.0, 1.0]))  # nan out: not comparable

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            norms(torch.zeros(4), torch.zeros(5))

    def test_one_rounding_costs_a_full_unit_not_half(self):
        """Pins the constant the whole budget table is built on.

        ``unit`` is the unit roundoff ``2**-precision``, already *half* the spacing between
        adjacent representables
        """
        midpoint = torch.tensor([1.0 + 2.0**-8], dtype=torch.float64)  # between 1 and 1+2**-7
        measured = rel_err_units(norms(midpoint.to(torch.bfloat16), midpoint), U_BF16)
        assert measured["max_rel_err_units"] == pytest.approx(1.0, abs=0.01)

    def test_global_normalization_lets_cast_error_grow_toward_one(self):
        """this checks 2 claims
        (1) <=1.0 -  a pure cast never exceeds one error unit (half adjacent representable spacing).
        (2) small < large, larger tensor has a higher chance hitting midpoint of spacing
        """
        with seeded(0):
            small = torch.randn(1024, dtype=torch.float64)
            large = torch.randn(1 << 20, dtype=torch.float64)

        def cast_units(r):
            return rel_err_units(norms(r.to(torch.bfloat16), r), U_BF16)["max_rel_err_units"]

        assert cast_units(small) < cast_units(large) <= 1.0


# ── 2. identity and seeding are reproducible ─────────────────────────────────


class TestDeterminism:
    def test_cell_id_is_stable_and_shape_order_independent(self):
        a = cell_id("aten::mm", {"self": (32, 64), "mat2": (64, 32)}, "bfloat16")
        b = cell_id("aten::mm", {"mat2": (64, 32), "self": (32, 64)}, "bfloat16")
        assert a == b, "dict ordering must not change the id"
        assert len(a) == 16

    def test_cell_id_separates_dtype_and_shape(self):
        base = cell_id("aten::mm", {"self": (32, 64)}, "bfloat16")
        assert base != cell_id("aten::mm", {"self": (32, 64)}, "float32")
        assert base != cell_id("aten::mm", {"self": (32, 65)}, "bfloat16")

    def test_same_cell_yields_identical_inputs(self):
        """The property that makes a run reproducible under xdist."""
        seed = seed_for(cell_id("aten::sum", {"self": (128,)}, "bfloat16"))
        with seeded(seed):
            first = torch.randn(128, dtype=torch.bfloat16)
        with seeded(seed):
            second = torch.randn(128, dtype=torch.bfloat16)
        assert torch.equal(first, second)

    def test_different_cells_yield_different_inputs(self):
        s1 = seed_for(cell_id("aten::sum", {"self": (128,)}, "bfloat16"))
        s2 = seed_for(cell_id("aten::mean", {"self": (128,)}, "bfloat16"))
        assert s1 != s2

    def test_seeded_restores_the_global_stream(self):
        """Cells must not perturb each other's draws."""
        torch.manual_seed(999)
        expected = torch.randn(4)

        torch.manual_seed(999)
        with seeded(7):
            torch.randn(1000)
        assert torch.equal(torch.randn(4), expected)


# ── 3. budget arithmetic ─────────────────────────────────────────────────────


class TestBudget:
    def test_the_flat_budget_is_one_full_ulp(self):
        """``2.0`` units is the output format's machine epsilon, not an arbitrary constant."""
        assert ARITHMETIC_BUDGET_UNITS == 2.0
        assert ARITHMETIC_BUDGET_UNITS * U_BF16 == 2.0**-7
        assert ARITHMETIC_BUDGET_UNITS * U_FP32 == 2.0**-23

    @pytest.mark.parametrize("group", ["reduction", "linalg", "nn", "elementwise", "foreach"])
    def test_every_arithmetic_group_gets_the_same_budget(self, group):
        """No per-op or per-group arithmetic left: shape and reduction width cannot change it."""
        assert error_budget(group) == ARITHMETIC_BUDGET_UNITS

    def test_an_unknown_group_is_gated_rather_than_skipped(self):
        """The coverage the flat budget buys: ~270 ops that used to report ``no_budget``."""
        assert error_budget("unknown") == ARITHMETIC_BUDGET_UNITS
        assert error_budget("fft") == ARITHMETIC_BUDGET_UNITS

    def test_bitexact_groups_budget_zero(self):
        assert error_budget("shape") == 0.0
        assert error_budget("indexing") == 0.0

    def test_bitexactness_survives_an_ungated_format(self):
        """A ``view`` must reproduce its input at every dtype, so ``gated`` must not reach it."""
        assert error_budget("shape", gated=False) == 0.0
        assert error_budget("indexing", gated=False) == 0.0

    def test_random_group_is_excluded(self):
        assert error_budget("random") is None
        assert error_budget("random", gated=False) is None

    def test_ungated_format_gets_no_verdict(self):
        """fp32 and finer: ``u_acc == u_in``, so accumulation not rounding sets the scale."""
        assert error_budget("reduction", gated=False) is None
        assert error_budget("nn", gated=False) is None

    def test_a_group_allowance_is_one_table_entry(self, monkeypatch):
        """The extension seam: a family measured above one ULP needs an entry, not a formula."""
        monkeypatch.setitem(budgets.GROUP_BUDGET_UNITS, "nn", 4.0)
        assert error_budget("nn") == 4.0
        assert error_budget("reduction") == ARITHMETIC_BUDGET_UNITS


# ── 4. THE DISCRIMINATOR: the measured error infers the accumulator ──────────


class TestAccumulatorDiscriminator:
    """Sum one bf16 tensor two ways on CPU and confirm the measurement tells them apart.

    Inputs are all-positive (uniform in [0,1))
    (1) the reference sum is then scaled at ~n/2, and same-sign rounding errors do not cancel.
    (2) we assert a *lower* bound on the bf16 error rather than claiming it matches sqrt(n).
    """

    N = 8192

    @staticmethod
    def _bf16_sequential_sum(x: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros((), dtype=torch.bfloat16)
        for v in x:
            acc = acc + v  # bf16 + bf16 stays bf16: one rounding per step
        return acc

    def _measure(self, actual: torch.Tensor, ref: torch.Tensor) -> float:
        return rel_err_units(norms(actual.double(), ref), BF16.unit)["max_rel_err_units"]

    @classmethod
    @pytest.fixture(scope="class")
    def data(cls):
        with seeded(seed_for(cell_id("aten::sum", {"self": (cls.N,)}, "bfloat16"))):
            x = torch.rand(cls.N, dtype=torch.bfloat16)
        return x, x.double().sum()

    def test_fp32_accumulation_passes_the_flat_gate(self, data):
        """Why the flat constant can afford to drop the ``u_acc`` term.

        With an fp32 accumulator the whole width contribution at n=8192 is
        ``sqrt(8192) * 2**-16 = 0.0014`` units, so a correct reduction is output-cast bound
        and lands well inside one full ULP whatever ``n`` is.
        """
        x, ref = data
        measured = self._measure(x.float().sum().to(torch.bfloat16), ref)
        assert (
            measured <= ARITHMETIC_BUDGET_UNITS
        ), f"fp32 accumulation should be output-cast bound, got {measured}"

    def test_bf16_accumulation_blows_past_the_flat_gate(self, data):
        """And why dropping it is safe: a narrow accumulator is not a near-miss.

        sqrt(8192) = 90.5 units against a 2.0 gate, so 25x is a slack threshold this should
        comfortably pass. The failure mode the ``u_acc`` term existed to predict is caught by
        the flat bound anyway -- it just is not attributed.
        """
        x, ref = data
        measured = self._measure(self._bf16_sequential_sum(x), ref)
        assert (
            measured > 25 * ARITHMETIC_BUDGET_UNITS
        ), f"bf16 accumulation at n={self.N} should be unmistakable, got {measured} units"

    def test_the_two_accumulators_separate_by_orders_of_magnitude(self, data):
        """The property the suite's diagnostic value rests on."""
        x, ref = data
        fp32_units = self._measure(x.float().sum().to(torch.bfloat16), ref)
        bf16_units = self._measure(self._bf16_sequential_sum(x), ref)
        assert bf16_units / max(fp32_units, U_BF16) > 10


# ── format plumbing ──────────────────────────────────────────────────────────


class TestFormats:
    def test_units_match_the_dtype_table(self):
        assert BF16.unit == 2.0**-8
        assert FP32.unit == 2.0**-24

    def test_lookup_by_dtype_and_by_name(self):
        assert get_format(torch.bfloat16) is get_format("bfloat16") is BF16

    def test_unknown_dtype_raises(self):
        """Covers the dtypes Neuron does not have: fp16, fp8, mxfp4 are all absent."""
        with pytest.raises(KeyError, match="no NumericFormat"):
            get_format("float128")
        with pytest.raises(KeyError, match="no NumericFormat"):
            get_format(torch.float16)

    def test_reference_upcasts_without_changing_values(self):
        """The ordering that keeps the cast out of the measurement."""

        class _Sample:
            def __init__(self, x):
                self.input, self.args, self.kwargs = x, (), {}

        x = torch.rand(16, dtype=torch.bfloat16)
        ref = BF16.reference(lambda t: t * 1.0, _Sample(x))
        assert ref.dtype == torch.float32
        assert torch.equal(ref, x.float()), "upcast must be exact, not a re-draw"

    def test_reference_leaves_integer_tensors_alone(self):
        """Casting an index tensor to fp32 would change dispatch."""

        class _Sample:
            def __init__(self, x, idx):
                self.input, self.args, self.kwargs = x, (idx,), {}

        idx = torch.tensor([0, 2, 1])
        out = BF16.reference(lambda t, i: t.index_select(0, i), _Sample(torch.rand(3, 4), idx))
        assert out.dtype == torch.float32


class TestCsvCell:
    """``_cell``: how a row value becomes one CSV cell.

    The nullable columns are the point. A cell that reads as empty means "not measured"; a
    cell holding the string ``"None"`` reads as a value, so every downstream numeric parse
    of ``budget_units`` / ``n`` / ``max_rel_err_units`` has to special-case it.
    """

    def test_none_becomes_empty_not_the_string_none(self):
        assert _cell(None) == ""

    def test_zero_and_false_survive(self):
        """``0`` is a real ``n`` (bit-exact group) and a real budget, not a missing value."""
        assert _cell(0) == "0"
        assert _cell(0.0) == "0.0"
        assert _cell(False) == "False"

    def test_newlines_are_flattened_so_one_row_stays_one_line(self):
        assert _cell("boom\nat line 2\r\n") == "boom at line 2 "

    def test_floats_keep_full_precision(self):
        """Error figures are compared against budgets downstream; rounding here would lie."""
        assert _cell(0.5506040337445887) == "0.5506040337445887"


# ── frobenius ratio + per-dtype reference ─────────────────────────────────


class TestFroRatio:
    """frobenius_ratio = ||trn_low - ref_high|| / ||cpu_low - ref_high|| (low = candidate dtype,
    high = next precision up: bf16 -> fp32, fp32 -> fp64)."""

    def test_ratio_is_candidate_over_baseline(self):
        assert frobenius_ratio(2.0, 1.0) == 2.0
        assert frobenius_ratio(1.0, 1.0) == 1.0
        assert frobenius_ratio(0.5, 1.0) == 0.5

    @pytest.mark.parametrize(
        "num, denom, expected",
        [
            (1e3, 1e-3, 1e6),
            (1e-3, 1e3, 1e-6),
            (1e3, 1e3, 1.0),
            (1e-3, 1e-3, 1.0),
        ],
        ids=["big/small", "small/big", "big/big", "small/small"],
    )
    def test_numerator_denominator_magnitude_combinations(self, num, denom, expected):
        """num large/small x denom large/small -- ratio is scale-free (only their quotient)."""
        assert frobenius_ratio(num, denom) == pytest.approx(expected)

    def test_bitexact_baseline_has_no_scale(self):
        assert frobenius_ratio(0.0, 0.0) == 0.0  # device also exact -> ok
        assert frobenius_ratio(1e-9, 0.0) == math.inf  # device drifts off a perfect baseline

    def test_from_norms_matches_manual(self):
        ref = torch.full((4096,), 1.0, dtype=torch.float64)
        cand = ref + 2.0 * U_BF16
        base = ref + 1.0 * U_BF16
        num = norms(cand, ref)["frobenius_abs_err"]
        den = norms(base, ref)["frobenius_abs_err"]
        assert frobenius_ratio(num, den) == pytest.approx(2.0, rel=1e-6)

    def test_norms_now_carry_frobenius(self):
        n = norms(torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64))
        assert {"max_abs_err", "max_abs_ref", "frobenius_abs_err", "frobenius_abs_ref"} <= set(n)
        assert n["frobenius_abs_err"] == pytest.approx(2.0)  # ||[-1,-1,-1,-1]||_2 = 2

    def test_threshold_default(self):
        assert FROBENIUS_RATIO_THRESHOLD == 1.1

    def test_verdict_enum_values(self):
        # str subclass keeps the CSV cell human-readable ("within" etc.) while callers
        # can still compare against FrobeniusVerdict.WITHIN for typo-safety.
        assert FrobeniusVerdict.WITHIN.value == "within"
        assert FrobeniusVerdict.OVER.value == "over"
        assert FrobeniusVerdict.INVALID.value == "invalid"
        assert FrobeniusVerdict.WITHIN == "within"


class TestReferenceDtype:
    """Each format references the next precision up (bf16 -> fp32, fp32 -> fp64), so the
    frobenius-ratio denominator is non-zero even for fp32."""

    def test_reference_dtypes(self):
        assert BF16.reference_dtype == torch.float32
        assert FP32.reference_dtype == torch.float64

    def test_fp32_reference_runs_in_fp64(self):
        """A no-op fp32 reference would zero the frobenius-ratio denominator; fp64 keeps it real."""

        class _FakeSample:
            """Minimal stand-in for a SampleInput: reference() only reads input/args/kwargs."""

            def __init__(self, x):
                self.input, self.args, self.kwargs = x, (), {}

        x = torch.randn(2048, dtype=torch.float32)
        out = FP32.reference(lambda t: t.sum(), _FakeSample(x))
        assert out.dtype == torch.float64
