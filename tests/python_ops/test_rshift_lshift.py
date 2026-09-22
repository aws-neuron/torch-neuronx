"""Tests for operator-form bitwise shifts: aten::__rshift__ and aten::__lshift__.

These are the Python operator overloads (`x >> y`, `x << y`) of
`torch.bitwise_right_shift` / `torch.bitwise_left_shift`. Each has four
schema variants that all need a lowering, or the unregistered ones bounce
to CPU fallback:

    aten::__rshift__.Tensor(Tensor self, Tensor other)
    aten::__rshift__.Scalar(Tensor self, Scalar other)
    aten::__rshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out)
    aten::__rshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out)

CPU is the reference throughout, so these tests track upstream semantics
rather than hand-computed constants.
"""

import operator

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

# Signed dtypes whose full value range survives the int32 round-trip that
# neuronxcc's shift primitive requires.
SIGNED_DTYPES = [torch.int8, torch.int16, torch.int32]
INT_DTYPES = [*SIGNED_DTYPES, torch.int64]
ALL_INT_DTYPES = [*INT_DTYPES, torch.uint8]

# (self_dtype, other_dtype) pairs that promote to a *third* dtype on CPU.
# Values stay tiny so only the dtype of the result is under test.
PROMOTION_PAIRS = [
    (torch.int8, torch.int16),
    (torch.int8, torch.int32),
    (torch.int8, torch.int64),
    (torch.int16, torch.int32),
    (torch.int16, torch.int8),
    (torch.int32, torch.int8),
    (torch.uint8, torch.int8),
    (torch.uint8, torch.int16),
]

SHIFT_OPS = [
    pytest.param(operator.rshift, "__rshift__", id="rshift"),
    pytest.param(operator.lshift, "__lshift__", id="lshift"),
]


def _clamp_to(dtype, values):
    """Keep `values` representable in `dtype` so the CPU reference is meaningful."""
    info = torch.iinfo(dtype)
    return [max(info.min, min(info.max, v)) for v in values]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestShiftOps:
    """aten::__rshift__ / aten::__lshift__ against a CPU reference."""

    @pytest.fixture
    def device(self):
        return torch.device("neuron")

    # ── Tensor overload ──────────────────────────────────────────────

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
    def test_tensor_overload(self, device, op, name, dtype):
        """`x >> y` / `x << y` with two tensors, per dtype."""
        x_cpu = torch.tensor(_clamp_to(dtype, [0, 1, 8, 64]), dtype=dtype)
        y_cpu = torch.tensor([0, 1, 2, 3], dtype=dtype)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
    @pytest.mark.parametrize("shift", [0, 1, 3])
    def test_scalar_overload(self, device, op, name, dtype, shift):
        """`x >> k` / `x << k` with a Python int, per dtype."""
        x_cpu = torch.tensor(_clamp_to(dtype, [0, 1, 8, 64]), dtype=dtype)
        expected = op(x_cpu, shift)

        with track_neuron_ops():
            result = op(x_cpu.to(device), shift)
            assert_op_runs_on_neuron(f"aten::{name}.Scalar")

        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu(), expected)

    # ── out= overloads ───────────────────────────────────────────────
    #
    # These exist in the schema and must be registered. When they aren't,
    # PyTorch's fallback path writes the result into `self` as well as
    # `out` — see test_out_does_not_corrupt_self below.

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_tensor_out_overload(self, device, op, name):
        """The `.Tensor_out` overload writes `out` and returns it."""
        x_cpu = torch.tensor([8, 16, 32], dtype=torch.int32)
        y_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = op(x_cpu, y_cpu)

        out = torch.zeros(3, dtype=torch.int32, device=device)
        with track_neuron_ops():
            aten_op = getattr(torch.ops.aten, name)
            result = aten_op.Tensor_out(x_cpu.to(device), y_cpu.to(device), out=out)
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        torch.testing.assert_close(out.cpu(), expected)
        torch.testing.assert_close(result.cpu(), expected)
        assert result.data_ptr() == out.data_ptr(), "out= must alias the returned tensor"

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_scalar_out_overload(self, device, op, name):
        """The `.Scalar_out` overload writes `out` and returns it."""
        x_cpu = torch.tensor([8, 16, 32], dtype=torch.int32)
        expected = op(x_cpu, 2)

        out = torch.zeros(3, dtype=torch.int32, device=device)
        with track_neuron_ops():
            aten_op = getattr(torch.ops.aten, name)
            result = aten_op.Scalar_out(x_cpu.to(device), 2, out=out)
            assert_op_runs_on_neuron(f"aten::{name}.Scalar")

        torch.testing.assert_close(out.cpu(), expected)
        torch.testing.assert_close(result.cpu(), expected)
        assert result.data_ptr() == out.data_ptr(), "out= must alias the returned tensor"

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("variant", ["Tensor_out", "Scalar_out"])
    def test_out_does_not_corrupt_self(self, device, op, name, variant):
        """Regression: an out= shift must not write through to `self`.

        With the `_out` overloads unregistered, the fallback path aliased
        `self` to `out`, so `torch.ops.aten.__rshift__.Tensor_out(x, y, out=o)`
        left x == o instead of leaving x untouched.
        """
        x_cpu = torch.tensor([8, 16, 32], dtype=torch.int32)
        x = x_cpu.to(device)
        out = torch.zeros(3, dtype=torch.int32, device=device)

        aten_op = getattr(torch.ops.aten, name)
        if variant == "Tensor_out":
            aten_op.Tensor_out(
                x, torch.tensor([1, 2, 3], dtype=torch.int32, device=device), out=out
            )
        else:
            aten_op.Scalar_out(x, 2, out=out)
        torch.neuron.synchronize()

        torch.testing.assert_close(x.cpu(), x_cpu, msg="self was mutated by an out= shift")

    # ── Type promotion ───────────────────────────────────────────────

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("self_dtype,other_dtype", PROMOTION_PAIRS)
    def test_mixed_dtype_promotion(self, device, op, name, self_dtype, other_dtype):
        """Mixed-dtype shifts promote per torch.result_type, matching CPU.

        Regression: the shared helper used to hardcode the result dtype to
        `self.dtype`, so `int8 >> int64` returned int8 where CPU gives int64.
        """
        x_cpu = torch.tensor([16], dtype=self_dtype)
        y_cpu = torch.tensor([1], dtype=other_dtype)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        assert (
            result.dtype == expected.dtype
        ), f"{self_dtype} {name} {other_dtype}: expected {expected.dtype}, got {result.dtype}"
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize(
        "torch_op",
        [torch.bitwise_right_shift, torch.bitwise_left_shift],
        ids=["right", "left"],
    )
    @pytest.mark.parametrize("dtype", ALL_INT_DTYPES)
    def test_scalar_tensor_direction(self, device, torch_op, dtype):
        """A Python scalar as the *first* operand: `5 >> tensor`.

        aten::__rshift__ has no Scalar_Tensor overload, so this is only
        reachable through torch.bitwise_*_shift -- but it shares the helper
        this change touches, and it is the one branch that takes the
        `x = torch.full_like(y, x)` path, which had no coverage.

        Note this cannot distinguish the old `x.dtype` rule from
        `torch.result_type`: full_like() gives x the same dtype as y, so both
        agree here by construction. It is branch coverage, not a regression
        guard -- see test_scalar_shift_does_not_widen_dtype for the latter.
        """
        y_cpu = torch.tensor([0, 1, 2], dtype=dtype)
        expected = torch_op(64, y_cpu)

        result = torch_op(64, y_cpu.to(device))

        assert result.dtype == expected.dtype, (
            f"scalar {torch_op.__name__} {dtype}: expected {expected.dtype}," f" got {result.dtype}"
        )
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_scalar_shift_does_not_widen_dtype(self, device, op, name, dtype):
        """A Python-int shift keeps the tensor's dtype, including for 0-d input.

        Regression: the promoted dtype was computed after the helper wrapped the
        Python scalar in a tensor, which defaults to int64. For a 0-d input both
        operands then looked 0-d, so promotion widened the result to int64 —
        doubling its size and breaking the packed-int64 paths ("Source and
        destination must have same number of bytes" in `int64_tensor == 0`).
        """
        for x_cpu in (torch.tensor([8, 16], dtype=dtype), torch.tensor(8, dtype=dtype)):
            expected = op(x_cpu, 1)

            result = op(x_cpu.to(device), 1)

            assert result.dtype == expected.dtype == dtype, (
                f"{dtype} shifted by a Python int became {result.dtype}" f" (ndim={x_cpu.dim()})"
            )
            torch.testing.assert_close(result.cpu(), expected)

    def test_packed_int64_comparison_still_works(self, device):
        """Guard the packed-int64 path that the shift helper feeds.

        The int64 sign-extend in kernel.py builds the high word with
        torch.bitwise_right_shift(low, 31), so a dtype regression in the shift
        helper silently corrupts every int64 op. This is the cheapest canary.
        """
        t = torch.zeros((1, 2, 3), dtype=torch.int64, device=device)
        assert bool((t == 0).all().item())

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_bool_scalar_promotion(self, device, op, name):
        """A bool tensor shifted by a Python int promotes to int64 on CPU."""
        x_cpu = torch.tensor([True, False, True])
        expected = op(x_cpu, 1)

        with track_neuron_ops():
            result = op(x_cpu.to(device), 1)
            assert_op_runs_on_neuron(f"aten::{name}.Scalar")

        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu(), expected)

    # ── Value semantics ──────────────────────────────────────────────

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_rshift_negative_is_arithmetic(self, device, dtype):
        """Right-shifting a negative int sign-extends (arithmetic, not logical)."""
        x_cpu = torch.tensor(_clamp_to(dtype, [-1, -8, -127]), dtype=dtype)
        y_cpu = torch.tensor([1, 1, 1], dtype=dtype)
        expected = x_cpu >> y_cpu

        with track_neuron_ops():
            result = x_cpu.to(device) >> y_cpu.to(device)
            assert_op_runs_on_neuron("aten::__rshift__.Tensor")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_lshift_overflow_wraps(self, device, dtype):
        """Left-shifting past the dtype's width truncates the same way CPU does."""
        info = torch.iinfo(dtype)
        x_cpu = torch.tensor([info.max, info.max // 2, 3], dtype=dtype)
        y_cpu = torch.tensor([2, 2, 2], dtype=dtype)
        expected = x_cpu << y_cpu

        with track_neuron_ops():
            result = x_cpu.to(device) << y_cpu.to(device)
            assert_op_runs_on_neuron("aten::__lshift__.Tensor")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_shift_at_or_past_bitwidth(self, device, op, name, dtype):
        """Shifting by >= the dtype's bit width matches CPU (upstream #70904)."""
        bits = torch.iinfo(dtype).bits
        x_cpu = torch.tensor([-1, 0, 1], dtype=dtype)
        y_cpu = torch.tensor([bits, bits, bits], dtype=dtype)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        torch.testing.assert_close(result.cpu(), expected)

    # ── Shapes & layout ──────────────────────────────────────────────

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    @pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)])
    def test_shapes(self, device, op, name, shape):
        torch.manual_seed(0)
        x_cpu = torch.randint(1, 64, shape, dtype=torch.int32)
        y_cpu = torch.randint(0, 4, shape, dtype=torch.int32)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_broadcasting(self, device, op, name):
        x_cpu = torch.tensor([[64], [128]], dtype=torch.int32)
        y_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        assert result.shape == expected.shape
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_broadcasting_into_out(self, device, op, name):
        """Regression: a broadcasting shift writing an explicit `out=` buffer.

        This is the exact shape the `is_inplace` fix repaired, and it needs
        both halves at once. With `__rshift__` misclassified as in-place, the
        wrapper set ``out = self``, so the output buffer was sized from `self`
        ``(2,1)`` rather than the broadcast result ``(2,3)`` and the op died
        with ``output tensor output0 is too small (tensor size: 8, minimum
        size: 24)``. `test_broadcasting` broadcasts without `out=`, and
        `test_tensor_out_overload` passes `out=` at the operand shape, so
        neither reaches this path. Only the `.Tensor_out` overload can get
        here -- a Scalar shift never broadcasts.
        """
        x_cpu = torch.tensor([[64], [128]], dtype=torch.int32)
        y_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = op(x_cpu, y_cpu)
        assert expected.shape == (2, 3), "the reference must actually broadcast"

        x = x_cpu.to(device)
        out = torch.zeros(2, 3, dtype=torch.int32, device=device)
        with track_neuron_ops():
            aten_op = getattr(torch.ops.aten, name)
            result = aten_op.Tensor_out(x, y_cpu.to(device), out=out)
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        torch.testing.assert_close(out.cpu(), expected)
        torch.testing.assert_close(result.cpu(), expected)
        assert result.data_ptr() == out.data_ptr(), "out= must alias the returned tensor"
        # `self` is the broadcast operand here, so a write-through would also
        # be a shape change -- assert it survived untouched.
        torch.testing.assert_close(x.cpu(), x_cpu, msg="self was mutated by an out= shift")

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_non_contiguous_input(self, device, op, name):
        """A transposed (non-contiguous) operand still matches CPU."""
        base_cpu = torch.arange(1, 13, dtype=torch.int32).reshape(3, 4)
        x_cpu = base_cpu.t()
        assert not x_cpu.is_contiguous()
        expected = op(x_cpu, 1)

        with track_neuron_ops():
            x = base_cpu.to(device).t()
            result = op(x, 1)
            assert_op_runs_on_neuron(f"aten::{name}.Scalar")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_zero_element(self, device, op, name):
        x_cpu = torch.empty(0, dtype=torch.int32)
        expected = op(x_cpu, x_cpu)

        result = op(x_cpu.to(device), x_cpu.to(device))

        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_scalar_tensor_0d(self, device, op, name):
        """A 0-d tensor operand broadcasts against a 1-d one."""
        x_cpu = torch.tensor([8, 16, 32], dtype=torch.int32)
        y_cpu = torch.tensor(1, dtype=torch.int32)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        torch.testing.assert_close(result.cpu(), expected)

    # ── int64 ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_int64_within_32bit_range(self, device, op, name):
        """int64 shifts are exact while the values fit in 32 bits."""
        x_cpu = torch.tensor([8, 256, 1024], dtype=torch.int64)
        y_cpu = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected = op(x_cpu, y_cpu)

        with track_neuron_ops():
            result = op(x_cpu.to(device), y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        assert result.dtype == torch.int64
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Platform limitation: Neuron truncates int64 to 32 bits, so bits above"
            " 31 are lost. Not shift-specific — int64 mul and bitwise_and truncate"
            " identically. Remove this xfail if 64-bit integer support lands."
        ),
    )
    def test_int64_beyond_32_bits(self, device):
        """Documents the int64 truncation limit; strict xfail flags a platform fix."""
        x_cpu = torch.tensor([1], dtype=torch.int64)
        y_cpu = torch.tensor([40], dtype=torch.int64)
        expected = x_cpu << y_cpu

        result = x_cpu.to(device) << y_cpu.to(device)

        torch.testing.assert_close(result.cpu(), expected)

    # ── Dispatch / API surface ───────────────────────────────────────

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_tensor_method_form(self, device, op, name):
        """`x.__rshift__(y)` dispatches to the same lowering as `x >> y`."""
        x_cpu = torch.tensor([16, 32, 64], dtype=torch.int32)
        y_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
        expected = getattr(x_cpu, name)(y_cpu)

        with track_neuron_ops():
            result = getattr(x_cpu.to(device), name)(y_cpu.to(device))
            assert_op_runs_on_neuron(f"aten::{name}.Tensor")

        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("op,name", SHIFT_OPS)
    def test_does_not_fall_back_to_cpu(self, device, op, name):
        """Explicitly assert neither overload appears in the fallback list."""
        x = torch.tensor([8, 16, 32], dtype=torch.int32, device=device)
        y = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)

        with track_neuron_ops():
            op(x, y)
            op(x, 2)
            torch.neuron.synchronize()
            fallbacks = torch_neuronx.get_fallback_ops()

        offenders = [o for o in fallbacks if name in o]
        assert not offenders, f"{name} fell back to CPU: {offenders}"
