"""Phase 1 acceptance tests: the error model, verified on CPU with no device.

TestAccumulatorDiscriminator validates ULP a bf16 accumulator differs from an fp32 one
by orders of magnitude.
"""

from __future__ import annotations

import math

import pytest
import torch

from .budgets import error_budget, width
from .conftest import _cell
from .formats import BF16, FP32, get_format
from .metrics import cell_id, norms, seed_for, seeded, to_ulp, ulps

U_BF16 = 2.0**-8
U_FP32 = 2.0**-24


# ── 1. a known injected error produces the expected ULP count ────────────────


class TestMetricArithmetic:
    def test_injected_error_reads_back_exactly(self):
        ref = torch.full((100,), 2.0, dtype=torch.float64)
        actual = ref.clone()
        actual[7] += 3.0 * U_BF16 * 2.0  # 3 ULP at the reference's scale

        n = norms(actual, ref)
        assert n["max_abs_ref"] == pytest.approx(2.0)
        assert to_ulp(n["max_abs_err"], n["max_abs_ref"], U_BF16) == pytest.approx(3.0)

    def test_exact_match_is_zero_ulp(self):
        ref = torch.randn(64, dtype=torch.float64)
        assert ulps(norms(ref.clone(), ref), U_BF16) == {"max_err_ulp": 0.0, "fro_err_ulp": 0.0}

    def test_zero_reference_does_not_silently_pass(self):
        """An all-zero reference has no scale. Exact -> 0 ULP; anything else -> inf."""
        zero = torch.zeros(8, dtype=torch.float64)
        assert to_ulp(0.0, 0.0, U_BF16) == 0.0
        assert to_ulp(1e-30, 0.0, U_BF16) == math.inf

        wrong = zero.clone()
        wrong[0] = 1e-30
        assert ulps(norms(wrong, zero), U_BF16)["max_err_ulp"] == math.inf

    def test_frobenius_and_linf_disagree_as_designed(self):
        """One bad element vs. broad drift -- each norm sees what the other misses."""
        ref = torch.full((4096,), 1.0, dtype=torch.float64)

        spike = ref.clone()
        spike[0] += 100.0 * U_BF16
        spike_u = ulps(norms(spike, ref), U_BF16)

        drift = ref + 2.0 * U_BF16
        drift_u = ulps(norms(drift, ref), U_BF16)

        assert spike_u["max_err_ulp"] > drift_u["max_err_ulp"]  # L-inf catches the spike
        assert drift_u["fro_err_ulp"] > spike_u["fro_err_ulp"]  # Frobenius catches the drift

    def test_nan_and_inf_outputs_are_distinguishable_from_the_metric_alone(self):
        inf = math.inf

        def measure(actual, ref):
            kw = {"dtype": torch.float64}
            n = norms(torch.tensor(actual, **kw), torch.tensor(ref, **kw))
            return to_ulp(n["max_abs_err"], n["max_abs_ref"], U_BF16)

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
        measured = ulps(norms(midpoint.to(torch.bfloat16), midpoint), U_BF16)["max_err_ulp"]
        assert measured == pytest.approx(1.0, abs=0.01)

    def test_global_normalization_lets_cast_error_grow_toward_one(self):
        """this checks 2 claims
        (1) <=1.0 -  a pure cast never exceeds one error unit (half adjacent representable spacing).
        (2) small < large, larger tensor has a higher chance hitting midpoint of spacing
        """
        with seeded(0):
            small = torch.randn(1024, dtype=torch.float64)
            large = torch.randn(1 << 20, dtype=torch.float64)

        def cast_ulp(r):
            return ulps(norms(r.to(torch.bfloat16), r), U_BF16)["max_err_ulp"]

        assert cast_ulp(small) < cast_ulp(large) <= 1.0


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


# ── 3. width extraction, the three cases that break on an assumed dim ────────


class TestWidth:
    def test_reduction_with_explicit_dim(self):
        assert (
            width("aten::sum.dim_IntList", "reduction", {"self": (128, 4096), "dim": [1]}) == 4096
        )
        assert width("aten::sum.dim_IntList", "reduction", {"self": (128, 4096), "dim": [0]}) == 128

    def test_reduction_without_dim_reduces_everything(self):
        """Mirrors ``_schema_make_sample``: a missing optional ``dim`` is reduce-all, not axis 0."""
        assert width("aten::sum", "reduction", {"self": (128, 4096)}) == 128 * 4096

    def test_reduction_with_negative_dim(self):
        assert width("aten::mean.dim", "reduction", {"self": (8, 16, 32), "dim": [-1]}) == 32

    def test_matmul_takes_the_contracted_dim(self):
        assert width("aten::mm", "linalg", {"self": (512, 2048), "mat2": (2048, 1024)}) == 2048
        assert (
            width("aten::bmm", "linalg", {"self": (4, 512, 2048), "mat2": (4, 2048, 1024)}) == 2048
        )

    def test_linear_contracts_the_weights_last_axis(self):
        """``linear`` stores weight as (out, in) -- the opposite of mm's (K, N)."""
        assert (
            width("aten::linear", "linalg", {"input": (1, 2048, 4096), "weight": (11008, 4096)})
            == 4096
        )

    def test_softmax_reads_the_axis_it_is_given(self):
        assert width("aten::_softmax", "nn", {"self": (32, 128, 8192), "dim": 1}) == 128
        assert width("aten::_softmax", "nn", {"self": (32, 128, 8192), "dim": -1}) == 8192

    @pytest.mark.parametrize("dim", [None, "<absent>"], ids=["null", "absent"])
    def test_softmax_without_a_usable_dim_is_unmodelled(self, dim):
        """``aten::_softmax`` takes a *required* ``int dim``, both cases are malformed."""
        shapes = {"self": (32, 128, 8192)}
        if dim != "<absent>":
            shapes["dim"] = dim
        assert width("aten::_softmax", "nn", shapes) is None
        assert error_budget("aten::_softmax", "nn", shapes, U_BF16, U_FP32) is None

    def test_null_dim_on_a_reduction_stays_reduce_all(self):
        """The deliberate divergence from ``_width_axis``: here null really means every axis."""
        shapes = {"self": (32, 128, 8192), "dim": None}
        assert width("aten::sum", "reduction", shapes) == 32 * 128 * 8192

    def test_layernorm_uses_normalized_shape(self):
        shapes = {"input": (2, 512, 4096), "normalized_shape": [4096]}
        assert width("aten::native_layer_norm", "nn", shapes) == 4096

    def test_elementwise_is_width_one(self):
        assert width("aten::add.Tensor", "elementwise", {"self": (4096, 4096)}) == 1

    def test_explicit_model_outranks_the_group_taxonomy(self):
        """
        Hardcodes ops group type to `elementwise`, assert n is resolved from ops count.
        note "baddbmm" isn't elementwise in fact, hardcode makes it go this path.
        """
        linear_shapes = {"input": (1, 2048, 4096), "weight": (11008, 4096)}
        assert width("aten::linear", "elementwise", linear_shapes) == 4096
        assert (
            width(
                "aten::baddbmm.dtype_out",
                "elementwise",
                {"self": (8, 256, 64), "batch1": (8, 256, 128), "batch2": (8, 128, 64)},
            )
            == 128
        )

    def test_unmodelled_op_returns_none(self):
        """``None`` is a reported non-verdict, not a crash."""
        assert width("aten::some_new_op", "unknown", {"self": (128,)}) is None

    def test_dict_form_shape_spec(self):
        """Registry entries may use the ``_tensor_from_spec`` dict form."""
        shapes = {"self": {"shape": (128, 4096), "dtype": "bfloat16"}, "dim": [1]}
        assert width("aten::sum.dim_IntList", "reduction", shapes) == 4096


# ── budget arithmetic ────────────────────────────────────────────────────────


class TestBudget:
    def test_accumulator_dominates_the_answer(self):
        """The two hypotheses the suite exists to distinguish, at n=4096."""
        shapes = {"self": (4096,)}
        fp32_acc = error_budget("aten::sum", "reduction", shapes, U_BF16, U_FP32)
        bf16_acc = error_budget("aten::sum", "reduction", shapes, U_BF16, U_BF16)

        assert fp32_acc == pytest.approx(1.0 + 64 / 65536, rel=1e-3)
        assert bf16_acc == pytest.approx(1.0 + 64, rel=1e-3)
        assert bf16_acc / fp32_acc > 60

    def test_fp32_accumulator_flattens_the_shape_dependence(self):
        small = error_budget("aten::sum", "reduction", {"self": (10,)}, U_BF16, U_FP32)
        large = error_budget("aten::sum", "reduction", {"self": (8192,)}, U_BF16, U_FP32)
        assert large / small < 1.01, "with fp32 accumulation, width should barely matter"

    def test_pairwise_is_much_tighter_than_sequential(self):
        shapes = {"self": (8192,)}
        seq = error_budget("aten::sum", "reduction", shapes, U_BF16, U_BF16, order="sequential")
        pair = error_budget("aten::sum", "reduction", shapes, U_BF16, U_BF16, order="pairwise")
        assert seq == pytest.approx(1.0 + math.sqrt(8192), rel=1e-3)
        assert pair == pytest.approx(1.0 + math.sqrt(13), rel=1e-2)

    def test_rms_norm_is_one_pass_not_two(self):
        """``x / sqrt(mean(x^2) + eps) * w`` is a single reduction.

        layer_norm's mean-then-variance is the two-pass case; giving rms_norm the same
        coefficient doubled its allowance for arithmetic it does not perform.
        """
        shapes = {"input": (2, 512, 4096), "normalized_shape": [4096]}
        rms = error_budget("aten::rms_norm", "nn", shapes, U_BF16, U_BF16)
        ln = error_budget("aten::layer_norm", "nn", shapes, U_BF16, U_BF16)
        assert rms == pytest.approx(2.0 + math.sqrt(4096), rel=1e-3)
        assert (ln - 2.0) == pytest.approx(2 * (rms - 2.0), rel=1e-3)

    def test_norm_layers_budget_two_roundings_not_one(self):
        """A norm layer rounds twice outside the reduction: scale-and-shift, then output cast.

        Under fp32 accumulation the width term all but vanishes, so this constant is the whole
        budget -- and at 1.0 it rejected every layer_norm cell measured on trn2 (1.29-1.80).
        """
        shapes = {"input": (8, 4096), "normalized_shape": [4096]}
        assert error_budget("aten::layer_norm", "nn", shapes, U_BF16, U_FP32) == pytest.approx(
            2.0, abs=0.01
        )

    def test_bitexact_groups_budget_zero(self):
        assert error_budget("aten::view", "shape", {"self": (4, 4)}, U_BF16, U_FP32) == 0.0
        assert (
            error_budget("aten::index_select", "indexing", {"self": (4, 4)}, U_BF16, U_FP32) == 0.0
        )

    def test_random_group_is_excluded(self):
        assert error_budget("aten::rand", "random", {"self": (4,)}, U_BF16, U_FP32) is None

    def test_unmodelled_op_has_no_budget(self):
        assert error_budget("aten::some_new_op", "unknown", {"self": (4,)}, U_BF16, U_FP32) is None

    def test_unknown_order_is_an_error_not_a_default(self):
        with pytest.raises(ValueError, match="unknown order"):
            error_budget("aten::sum", "reduction", {"self": (8,)}, U_BF16, U_FP32, order="magic")


# ── 4. THE DISCRIMINATOR: measured ULP infers the accumulator ────────────────


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
        return ulps(norms(actual.double(), ref), BF16.unit)["max_err_ulp"]

    @classmethod
    @pytest.fixture(scope="class")
    def data(cls):
        with seeded(seed_for(cell_id("aten::sum", {"self": (cls.N,)}, "bfloat16"))):
            x = torch.rand(cls.N, dtype=torch.bfloat16)
        return x, x.double().sum()

    def test_fp32_accumulation_lands_within_one_ulp(self, data):
        x, ref = data
        measured = self._measure(x.float().sum().to(torch.bfloat16), ref)
        budget = error_budget("aten::sum", "reduction", {"self": (self.N,)}, U_BF16, U_FP32)
        assert budget == pytest.approx(1.0, abs=0.01), "fp32 accumulation leaves only the cast"
        assert measured <= budget, f"fp32 accumulation should be output-cast bound, got {measured}"

    def test_bf16_accumulation_blows_past_the_fp32_budget(self, data):
        """
        sqrt(8192) = 90.5, 50 is a slacked threshold that this should comfortably pass
        """
        x, ref = data
        measured = self._measure(self._bf16_sequential_sum(x), ref)
        fp32_budget = error_budget("aten::sum", "reduction", {"self": (self.N,)}, U_BF16, U_FP32)
        assert (
            measured > 50 * fp32_budget
        ), f"bf16 accumulation at n={self.N} should be unmistakable, got {measured} ULP"

    def test_the_two_accumulators_separate_by_orders_of_magnitude(self, data):
        """The property the suite's diagnostic value rests on."""
        x, ref = data
        fp32_ulp = self._measure(x.float().sum().to(torch.bfloat16), ref)
        bf16_ulp = self._measure(self._bf16_sequential_sum(x), ref)
        assert bf16_ulp / max(fp32_ulp, U_BF16) > 10


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
    of ``budget_ulp`` / ``n`` / ``max_err_ulp`` has to special-case it.
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
        """ULP figures are compared against budgets downstream; rounding here would lie."""
        assert _cell(0.5506040337445887) == "0.5506040337445887"
