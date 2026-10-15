"""Eager tests for the aten upsample/interpolate family on Neuron.

Covers the 14 op packets lowered in ``python_ops/torch_mlir/ops/interpolation.py``:

    upsample_nearest{1,2}d              _upsample_nearest_exact{1,2}d
    upsample_linear1d                   upsample_bilinear2d
    upsample_bicubic2d                  ... and the backward of each

CPU is the reference throughout, so these tests track upstream behaviour rather
than hand-written constants. Three aten quirks are pinned explicitly because a
rank-generic implementation gets them wrong (all three verified against CPU in
float64 over a full size x scale sweep):

1. ``upsample_nearest1d`` forward honours an explicit ``scales`` even when
   ``out == in``, while ``upsample_nearest2d`` forward short-circuits
   ``out == in`` and ``out == 2*in`` and ignores the scale.
2. the linear family short-circuits **per dimension** whose size is unchanged,
   dropping only that dimension's scale.
3. ``upsample_bicubic2d_backward`` copies when *both* spatial sizes are
   unchanged, ignoring the scales, while its own forward never short-circuits —
   so there the backward is not the adjoint of the forward.
"""

from types import SimpleNamespace

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

aten = torch.ops.aten

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# ── configuration matrices ───────────────────────────────────────────────────
# (in_w, out_w, scales)
NEAREST_1D_CFGS = [
    (4, 8, None),  # exact 2x upsample
    (4, 4, None),  # identity
    (4, 4, 2.0),  # out == in WITH a scale: the 1-D forward honours it (quirk 1)
    (4, 8, 3.0),  # out == 2*in WITH a scale: 1-D honours it, 2-D would not
    (8, 3, None),  # downsample
    (5, 7, None),  # non-integer ratio
    (1, 4, None),  # size-1 source
    (4, 1, None),  # collapse to one
    (3, 6, 2.0),  # scale agreeing with the size ratio
]

# ((in_h, in_w), (out_h, out_w), (scales_h, scales_w))
NEAREST_2D_CFGS = [
    ((4, 4), (8, 8), (None, None)),  # 2x, hits the out == 2*in fast path
    ((4, 4), (4, 4), (2.0, 2.0)),  # fast path ignores the scale (quirk 1)
    ((4, 4), (8, 8), (3.0, 3.0)),  # out == 2*in fast path ignores the scale
    ((4, 6), (2, 3), (None, None)),  # downsample
    ((5, 3), (7, 4), (None, None)),  # non-integer ratio, non-square
    ((4, 4), (4, 7), (2.0, None)),  # H unchanged (scale dropped), W resampled
    ((1, 4), (3, 8), (None, None)),  # size-1 spatial dim
]

LINEAR_1D_CFGS = [
    (4, 8, None),
    (4, 4, None),
    (4, 4, 2.0),  # per-dim short-circuit: identity, scale ignored (quirk 2)
    (8, 3, None),
    (5, 7, None),
    (1, 4, None),
    (4, 1, None),
    (3, 9, 3.0),
]

BIMODE_2D_CFGS = [
    ((4, 4), (8, 8), (None, None)),
    ((4, 4), (4, 4), (2.0, 2.0)),  # both dims short-circuit
    ((4, 5), (4, 9), (2.0, None)),  # H short-circuits, W does not (quirk 2)
    ((6, 6), (3, 2), (None, None)),  # downsample
    ((5, 3), (7, 4), (None, None)),
    ((2, 2), (5, 5), (None, None)),  # small source, bicubic taps clamp at edges
    ((1, 4), (3, 8), (None, None)),
]

NEAREST_OPS_1D = [("aten::upsample_nearest1d", False), ("aten::_upsample_nearest_exact1d", True)]
NEAREST_OPS_2D = [("aten::upsample_nearest2d", False), ("aten::_upsample_nearest_exact2d", True)]


def _nearest_1d(exact):
    return aten._upsample_nearest_exact1d if exact else aten.upsample_nearest1d


def _nearest_2d(exact):
    return aten._upsample_nearest_exact2d if exact else aten.upsample_nearest2d


def _nearest_1d_backward(exact):
    return aten._upsample_nearest_exact1d_backward if exact else aten.upsample_nearest1d_backward


def _nearest_2d_backward(exact):
    return aten._upsample_nearest_exact2d_backward if exact else aten.upsample_nearest2d_backward


def _assert_ran_on_device(op):
    """Assert the op's packet executed on Neuron rather than falling back.

    Derives the packet name from the OpOverload/OpOverloadPacket so parametrized
    tests can assert without repeating the name. ``assert_op_runs_on_neuron``
    substring-matches, so the packet name also covers the structured ``.out`` /
    ``.grad_input`` kernel the backend actually reports.
    """
    packet = str(op).removeprefix("aten.").split(".")[0]
    assert_op_runs_on_neuron(f"aten::{packet}")


# ═════════════════════════ nearest forward ═════════════════════════
class TestNearestForward:
    """Nearest neighbour only *moves* values, so every check is bit-exact."""

    @pytest.mark.parametrize("op_name,exact", NEAREST_OPS_1D)
    @pytest.mark.parametrize("in_w,out_w,scales", NEAREST_1D_CFGS)
    def test_1d_values(self, op_name, exact, in_w, out_w, scales):
        op = _nearest_1d(exact)
        x = torch.randn(2, 3, in_w)
        expected = op(x, [out_w], scales)
        with track_neuron_ops():
            got = op(x.to("neuron"), [out_w], scales)
            assert_op_runs_on_neuron(op_name)
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("op_name,exact", NEAREST_OPS_2D)
    @pytest.mark.parametrize("in_hw,out_hw,scales", NEAREST_2D_CFGS)
    def test_2d_values(self, op_name, exact, in_hw, out_hw, scales):
        op = _nearest_2d(exact)
        x = torch.randn(2, 3, *in_hw)
        expected = op(x, list(out_hw), *scales)
        with track_neuron_ops():
            got = op(x.to("neuron"), list(out_hw), *scales)
            assert_op_runs_on_neuron(op_name)
        assert torch.equal(got.cpu(), expected)

    def test_1d_honours_scale_when_size_unchanged(self):
        """Quirk 1, stated as its own test: 1-D honours the scale, 2-D does not.

        With ``in == out == 4`` and ``scales=2.0`` the 1-D forward samples
        ``floor(dst * 0.5)`` = [0,0,1,1] while the 2-D forward short-circuits to
        the identity. A rank-generic implementation gets one of the two wrong.
        """
        x = torch.randn(1, 1, 4)
        expected_1d = aten.upsample_nearest1d(x, [4], 2.0)
        assert torch.equal(expected_1d, x[..., [0, 0, 1, 1]]), "CPU reference changed"
        with track_neuron_ops():
            got = aten.upsample_nearest1d(x.to("neuron"), [4], 2.0)
            assert_op_runs_on_neuron("aten::upsample_nearest1d")
        assert torch.equal(got.cpu(), expected_1d)

        x2 = torch.randn(1, 1, 4, 4)
        expected_2d = aten.upsample_nearest2d(x2, [4, 4], 2.0, 2.0)
        assert torch.equal(expected_2d, x2), "CPU 2-D reference should be the identity"
        with track_neuron_ops():
            got2 = aten.upsample_nearest2d(x2.to("neuron"), [4, 4], 2.0, 2.0)
            assert_op_runs_on_neuron("aten::upsample_nearest2d")
        assert torch.equal(got2.cpu(), expected_2d)

    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("dtype", [*FLOAT_DTYPES, torch.uint8])
    def test_2d_dtypes(self, exact, dtype):
        op = _nearest_2d(exact)
        if dtype == torch.uint8:
            x = torch.randint(0, 256, (2, 3, 4, 5), dtype=dtype)
        else:
            x = torch.randn(2, 3, 4, 5, dtype=dtype)
        expected = op(x, [7, 9])
        with track_neuron_ops():
            got = op(x.to("neuron"), [7, 9])
            _assert_ran_on_device(op)
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("dtype", [*FLOAT_DTYPES, torch.uint8])
    def test_1d_dtypes(self, exact, dtype):
        op = _nearest_1d(exact)
        if dtype == torch.uint8:
            x = torch.randint(0, 256, (2, 3, 5), dtype=dtype)
        else:
            x = torch.randn(2, 3, 5, dtype=dtype)
        expected = op(x, [8])
        with track_neuron_ops():
            got = op(x.to("neuron"), [8])
            _assert_ran_on_device(op)
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), expected)


# ═════════════════════════ linear family forward ═════════════════════════
class TestLinearFamilyForward:
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_w,out_w,scales", LINEAR_1D_CFGS)
    def test_linear1d_values(self, align_corners, in_w, out_w, scales):
        x = torch.randn(2, 3, in_w)
        expected = aten.upsample_linear1d(x, [out_w], align_corners, scales)
        with track_neuron_ops():
            got = aten.upsample_linear1d(x.to("neuron"), [out_w], align_corners, scales)
            assert_op_runs_on_neuron("aten::upsample_linear1d")
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_hw,out_hw,scales", BIMODE_2D_CFGS)
    def test_bimode2d_values(self, mode, align_corners, in_hw, out_hw, scales):
        op = aten.upsample_bilinear2d if mode == "bilinear" else aten.upsample_bicubic2d
        op_name = f"aten::upsample_{mode}2d" if mode == "bilinear" else "aten::upsample_bicubic2d"
        x = torch.randn(2, 3, *in_hw)
        expected = op(x, list(out_hw), align_corners, *scales)
        with track_neuron_ops():
            got = op(x.to("neuron"), list(out_hw), align_corners, *scales)
            assert_op_runs_on_neuron(op_name)
        torch.testing.assert_close(got.cpu(), expected)

    def test_per_dimension_short_circuit(self):
        """Quirk 2: only the unchanged dimension drops its scale.

        ``(4,5) -> (4,9)`` with ``scales=(2.0, None)``: H is unchanged so aten
        ignores ``scales_h`` and copies, while W still resamples 5 -> 9.
        """
        x = torch.randn(1, 2, 4, 5)
        expected = aten.upsample_bilinear2d(x, [4, 9], False, 2.0, None)
        # H really is untouched: every output row matches its input row exactly.
        torch.testing.assert_close(expected, aten.upsample_bilinear2d(x, [4, 9], False, None, None))
        with track_neuron_ops():
            got = aten.upsample_bilinear2d(x.to("neuron"), [4, 9], False, 2.0, None)
            assert_op_runs_on_neuron("aten::upsample_bilinear2d")
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_linear1d_dtypes(self, dtype):
        x = torch.randn(2, 3, 5, dtype=dtype)
        expected = aten.upsample_linear1d(x, [8], False)
        with track_neuron_ops():
            got = aten.upsample_linear1d(x.to("neuron"), [8], False)
            _assert_ran_on_device(aten.upsample_linear1d)
        assert got.dtype == dtype
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_bimode2d_dtypes(self, mode, dtype):
        op = aten.upsample_bilinear2d if mode == "bilinear" else aten.upsample_bicubic2d
        x = torch.randn(2, 3, 4, 5, dtype=dtype)
        expected = op(x, [7, 9], False)
        with track_neuron_ops():
            got = op(x.to("neuron"), [7, 9], False)
            _assert_ran_on_device(op)
        assert got.dtype == dtype
        tolerance = {}
        if mode == "bicubic" and dtype != torch.float32:
            # aten's bicubic accumulates its four 1-D coefficients in the input
            # dtype, while this lowering (like aten's own bilinear) accumulates in
            # fp32 and narrows once. Measured against aten's fp32 result, ours is
            # the more accurate of the two; the two disagree by 2 ULP of fp16 and
            # 1 ULP of bfloat16. Compared absolutely because a handful of outputs
            # land near zero, where any ULP-sized difference is huge in relative
            # terms.
            eps = torch.finfo(dtype).eps
            tolerance = {"atol": 4 * eps, "rtol": 4 * eps}
        torch.testing.assert_close(got.cpu(), expected, **tolerance)


# ═════════════════════════ uint8 resampling numerics ═════════════════════════
class TestUint8Resampling:
    """uint8 bilinear/bicubic must be resampled in fp32, then rounded + clamped.

    Materialising the tap weights in the *input* dtype rounds every coefficient
    to 0 or 1 for uint8 and produces garbage — the failure mode
    ``test_uint8_is_not_degenerate`` guards.

    The two aten uint8 kernels then differ in where they narrow, so the lowering
    has to as well: bicubic stores its intermediate 1-D coefficients in uint8
    mid-kernel (reproduced by narrowing between the two passes, which takes the
    max deviation from 16 to 1), while bilinear narrows only at the end.
    """

    @staticmethod
    def _assert_uint8_close(mode, got, expected):
        """Within 1 of the CPU uint8 result, with a small mean deviation.

        Reproducing aten's half-up rounding and its narrowing after each 1-D pass
        gets the integral path to bit-exact against aten on most shapes — exactly
        enough that upstream's own ``xfail(..., dtypes=(torch.uint8,))  # off by one
        error`` in test_decomp now *unexpectedly succeeds* for bilinear. It is not
        exact everywhere: the last bit of a device-computed float32 weight can flip
        a value sitting precisely on .5, which moves that element by one. Hence ≤ 1
        rather than equality — still far tighter than the statistical band
        ``test_upsamplingBiMode2d_consistency`` settles for, so a regression in the
        rounding or the narrowing fails here.
        """
        diff = (got.cpu().to(torch.int16) - expected.to(torch.int16)).abs().to(torch.float32)
        assert diff.max().item() <= 1, f"{mode}: max deviation {diff.max().item()} > 1"
        assert diff.mean().item() < 0.1, f"{mode}: mean deviation {diff.mean().item()} >= 0.1"

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_hw,out_hw", [((4, 4), (8, 8)), ((7, 5), (3, 9)), ((2, 2), (5, 5))])
    def test_uint8_matches_cpu(self, mode, align_corners, in_hw, out_hw):
        op = aten.upsample_bilinear2d if mode == "bilinear" else aten.upsample_bicubic2d
        x = torch.randint(0, 256, (2, 3, *in_hw), dtype=torch.uint8)
        expected = op(x, list(out_hw), align_corners)
        with track_neuron_ops():
            got = op(x.to("neuron"), list(out_hw), align_corners)
            _assert_ran_on_device(op)
        assert got.dtype == torch.uint8
        self._assert_uint8_close(mode, got, expected)

    def test_uint8_bicubic_saturates_on_overshoot(self):
        """bicubic rings outside [0, 255]; those values must saturate, not wrap.

        This is the case that exposes the range the saturation clamps against. The
        eager path widens uint8 to int32 before tracing, so a clamp derived from
        the traced tensor's own dtype uses the +/-2^31 range, does nothing, and the
        framework's int32 -> uint8 narrowing afterwards wraps: an undershoot of -5
        comes back as 251 and this test sees a deviation of ~255 instead of <= 1.
        """
        x = torch.zeros(1, 1, 6, 6, dtype=torch.uint8)
        x[..., 2:4, 2:4] = 255  # sharp edge -> bicubic over/undershoots
        expected = aten.upsample_bicubic2d(x, [12, 12], False)
        float_result = aten.upsample_bicubic2d(x.to(torch.float32), [12, 12], False)
        assert float_result.min() < 0 or float_result.max() > 255, "input does not overshoot"
        with track_neuron_ops():
            got = aten.upsample_bicubic2d(x.to("neuron"), [12, 12], False)
            _assert_ran_on_device(aten.upsample_bicubic2d)
        self._assert_uint8_close("bicubic", got, expected)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    def test_uint8_is_not_degenerate(self, mode):
        """Guard against the weights-in-input-dtype bug specifically.

        Rounding the tap weights to uint8 makes the result a *gather* of input
        pixels (or zeros), so it can no longer produce intermediate values. A
        smooth ramp upsampled 4x must contain values that appear nowhere in the
        input.
        """
        op = aten.upsample_bilinear2d if mode == "bilinear" else aten.upsample_bicubic2d
        x = (torch.arange(16, dtype=torch.uint8) * 16).reshape(1, 1, 4, 4)
        with track_neuron_ops():
            got = op(x.to("neuron"), [16, 16], False).cpu()
            _assert_ran_on_device(op)
        interpolated = set(got.flatten().tolist()) - set(x.flatten().tolist())
        assert interpolated, "no interpolated values produced — tap weights lost precision"

    def test_uint8_bilinear_edge_stays_in_range(self):
        """The same sharp edge through bilinear, which is a convex blend.

        It cannot leave [0, 255], so it passes even with a broken saturation — the
        contrast with the bicubic version above is what shows the saturation only
        matters for modes that ring.
        """
        x = torch.zeros(1, 1, 6, 6, dtype=torch.uint8)
        x[..., 2:4, 2:4] = 255  # sharp edge
        expected = aten.upsample_bilinear2d(x, [12, 12], False)
        with track_neuron_ops():
            got = aten.upsample_bilinear2d(x.to("neuron"), [12, 12], False)
            _assert_ran_on_device(aten.upsample_bilinear2d)
        assert got.dtype == torch.uint8
        self._assert_uint8_close("bilinear", got, expected)


# ═════════════════════════ nearest backward ═════════════════════════
class TestNearestBackward:
    """The nearest backward scatter-adds each output gradient to its source pixel.

    Payloads are small integers so the accumulation is exact in any order —
    ``index_add`` does not promise aten's serial summation order, and a float
    payload would make a bit-exact comparison illegitimate rather than strict.
    """

    @staticmethod
    def _int_payload(*shape):
        return torch.randint(-8, 9, shape).float()

    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("in_w,out_w,scales", NEAREST_1D_CFGS)
    def test_1d_values(self, exact, in_w, out_w, scales):
        op = _nearest_1d_backward(exact)
        grad = self._int_payload(2, 3, out_w)
        input_size = [2, 3, in_w]
        expected = op(grad, [out_w], input_size, scales)
        with track_neuron_ops():
            got = op(grad.to("neuron"), [out_w], input_size, scales)
            _assert_ran_on_device(op)
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("exact", [False, True])
    @pytest.mark.parametrize("in_hw,out_hw,scales", NEAREST_2D_CFGS)
    def test_2d_values(self, exact, in_hw, out_hw, scales):
        op = _nearest_2d_backward(exact)
        grad = self._int_payload(2, 3, *out_hw)
        input_size = [2, 3, *in_hw]
        expected = op(grad, list(out_hw), input_size, *scales)
        with track_neuron_ops():
            got = op(grad.to("neuron"), list(out_hw), input_size, *scales)
            _assert_ran_on_device(op)
        assert torch.equal(got.cpu(), expected)

    def test_1d_backward_ignores_scale_its_forward_honours(self):
        """The 1-D nearest forward and backward are genuinely not adjoint.

        With ``in == out == 4`` and ``scales=2.0`` the forward samples
        [0,0,1,1], so a naive adjoint would accumulate two gradients into each of
        the first two positions. aten's backward instead takes the fast path and
        is the identity. Reproducing the forward's map here would be wrong.
        """
        grad = self._int_payload(1, 1, 4)
        expected = aten.upsample_nearest1d_backward(grad, [4], [1, 1, 4], 2.0)
        assert torch.equal(expected, grad), "CPU backward should be the identity here"
        with track_neuron_ops():
            got = aten.upsample_nearest1d_backward(grad.to("neuron"), [4], [1, 1, 4], 2.0)
            assert_op_runs_on_neuron("aten::upsample_nearest1d_backward")
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_dtypes(self, dtype):
        grad = self._int_payload(2, 3, 8, 10).to(dtype)
        expected = aten.upsample_nearest2d_backward(grad, [8, 10], [2, 3, 4, 5])
        with track_neuron_ops():
            got = aten.upsample_nearest2d_backward(grad.to("neuron"), [8, 10], [2, 3, 4, 5])
            _assert_ran_on_device(aten.upsample_nearest2d_backward)
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), expected)


# ═════════════════════════ linear family backward ═════════════════════════
class TestLinearFamilyBackward:
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_w,out_w,scales", LINEAR_1D_CFGS)
    def test_linear1d_values(self, align_corners, in_w, out_w, scales):
        grad = torch.randn(2, 3, out_w)
        input_size = [2, 3, in_w]
        expected = aten.upsample_linear1d_backward(grad, [out_w], input_size, align_corners, scales)
        with track_neuron_ops():
            got = aten.upsample_linear1d_backward(
                grad.to("neuron"), [out_w], input_size, align_corners, scales
            )
            assert_op_runs_on_neuron("aten::upsample_linear1d_backward")
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_hw,out_hw,scales", BIMODE_2D_CFGS)
    def test_bimode2d_values(self, mode, align_corners, in_hw, out_hw, scales):
        op = (
            aten.upsample_bilinear2d_backward
            if mode == "bilinear"
            else aten.upsample_bicubic2d_backward
        )
        grad = torch.randn(2, 3, *out_hw)
        input_size = [2, 3, *in_hw]
        expected = op(grad, list(out_hw), input_size, align_corners, *scales)
        with track_neuron_ops():
            got = op(grad.to("neuron"), list(out_hw), input_size, align_corners, *scales)
            assert_op_runs_on_neuron(f"aten::upsample_{mode}2d_backward")
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("scales", [(None, 0.5), (2.0, None), (2.0, 0.5)])
    def test_bicubic_backward_copies_when_both_sizes_unchanged(self, scales):
        """Quirk 3: the copy short-circuit ignores scales the forward honours.

        Only observable with an explicit scale — without one the adjoint is the
        identity anyway.
        """
        grad = torch.randn(1, 2, 3, 3)
        input_size = [1, 2, 3, 3]
        expected = aten.upsample_bicubic2d_backward(grad, [3, 3], input_size, False, *scales)
        # Confirm the reference really is a copy, so the assertion below has teeth.
        torch.testing.assert_close(expected, grad)
        with track_neuron_ops():
            got = aten.upsample_bicubic2d_backward(
                grad.to("neuron"), [3, 3], input_size, False, *scales
            )
            _assert_ran_on_device(aten.upsample_bicubic2d_backward)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_bimode2d_dtypes(self, mode, dtype):
        op = (
            aten.upsample_bilinear2d_backward
            if mode == "bilinear"
            else aten.upsample_bicubic2d_backward
        )
        grad = torch.randn(2, 3, 8, 10, dtype=dtype)
        expected = op(grad, [8, 10], [2, 3, 4, 5], False)
        with track_neuron_ops():
            got = op(grad.to("neuron"), [8, 10], [2, 3, 4, 5], False)
            _assert_ran_on_device(op)
        assert got.dtype == dtype
        # The scatter accumulates in fp32 and narrows once, so the result can be
        # a shade more accurate than aten's in-dtype accumulation.
        tolerance = {}
        if dtype != torch.float32:
            slack = 8 * torch.finfo(dtype).eps
            tolerance = {"atol": slack, "rtol": slack}
        torch.testing.assert_close(got.cpu(), expected, **tolerance)

    @pytest.mark.parametrize(
        "op,grad_shape,args",
        [
            (aten.upsample_nearest1d_backward, (2, 3, 8), ([8], [2, 3, 5], None)),
            (aten.upsample_nearest2d_backward, (2, 3, 8, 10), ([8, 10], [2, 3, 4, 5], None, None)),
            (aten.upsample_linear1d_backward, (2, 3, 8), ([8], [2, 3, 5], False, None)),
            (
                aten.upsample_bilinear2d_backward,
                (2, 3, 8, 10),
                ([8, 10], [2, 3, 4, 5], False, None, None),
            ),
            (
                aten.upsample_bicubic2d_backward,
                (2, 3, 8, 10),
                ([8, 10], [2, 3, 4, 5], False, None, None),
            ),
        ],
    )
    def test_grad_input_overload(self, op, grad_shape, args):
        """Each backward's ``.grad_input`` variant writes the same values."""
        grad = torch.randn(*grad_shape)
        expected = op(grad, *args)
        grad_n = grad.to("neuron")
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = op.grad_input(grad_n, *args, grad_input=out_n)
            _assert_ran_on_device(op)
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(out_n.cpu(), expected)
        torch.testing.assert_close(grad_n.cpu(), grad)


# ═════════════════════════ autograd end to end ═════════════════════════
class TestAutograd:
    """``.backward()`` through each mode, which is what picks the backward op.

    A forward that lowers correctly can still be paired with the wrong backward
    (or a CPU-fallback one), and only going through autograd catches that.
    """

    @pytest.mark.parametrize(
        "mode,kwargs,rank",
        [
            ("nearest", {}, 1),
            ("nearest", {}, 2),
            ("nearest-exact", {}, 1),
            ("nearest-exact", {}, 2),
            ("linear", {"align_corners": False}, 1),
            ("linear", {"align_corners": True}, 1),
            ("bilinear", {"align_corners": False}, 2),
            ("bilinear", {"align_corners": True}, 2),
            ("bicubic", {"align_corners": False}, 2),
            ("bicubic", {"align_corners": True}, 2),
        ],
    )
    def test_interpolate_backward(self, mode, kwargs, rank):
        shape = (2, 3, 5) if rank == 1 else (2, 3, 4, 5)
        size = 9 if rank == 1 else (7, 9)
        x_cpu = torch.randn(*shape, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = torch.nn.functional.interpolate(x_cpu, size=size, mode=mode, **kwargs)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        with track_neuron_ops():
            out_n = torch.nn.functional.interpolate(x_n, size=size, mode=mode, **kwargs)
            out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(out_n.detach().cpu(), out_cpu.detach())
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)

    @pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
    def test_scale_factor_backward(self, mode):
        """``scale_factor`` routes the backward through the ``.vec`` overload."""
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        x_cpu = torch.randn(2, 3, 4, 5, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)

        out_cpu = torch.nn.functional.interpolate(x_cpu, scale_factor=2.0, mode=mode, **kwargs)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)

        with track_neuron_ops():
            out_n = torch.nn.functional.interpolate(x_n, scale_factor=2.0, mode=mode, **kwargs)
            out_n.backward(grad.to("neuron"))

        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)

    def test_upsample_module_backward(self):
        module = torch.nn.Upsample(size=(7, 9), mode="bilinear", align_corners=False)
        x_cpu = torch.randn(2, 3, 4, 5, requires_grad=True)
        x_n = x_cpu.detach().clone().to("neuron").requires_grad_(True)
        out_cpu = module(x_cpu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)
        with track_neuron_ops():
            out_n = module(x_n)
            out_n.backward(grad.to("neuron"))
        torch.testing.assert_close(x_n.grad.cpu(), x_cpu.grad)


# ═════════════════════════ overloads ═════════════════════════
class TestOverloads:
    """Every overload in each schema is registered, including ``vec_out``."""

    @pytest.mark.parametrize(
        "op,args",
        [
            (aten.upsample_nearest1d, ([8], None)),
            (aten._upsample_nearest_exact1d, ([8], None)),
            (aten.upsample_linear1d, ([8], False, None)),
            (aten.upsample_nearest2d, ([7, 9], None, None)),
            (aten._upsample_nearest_exact2d, ([7, 9], None, None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None, None)),
            (aten.upsample_bicubic2d, ([7, 9], False, None, None)),
        ],
    )
    def test_out_overload(self, op, args):
        rank = 1 if "1d" in str(op) else 2
        x = torch.randn(2, 3, 5) if rank == 1 else torch.randn(2, 3, 4, 5)
        expected = op(x, *args)
        x_n = x.to("neuron")
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = op.out(x_n, *args, out=out_n)
            _assert_ran_on_device(op)
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(out_n.cpu(), expected)
        # out= must not disturb self (the fallback path aliases them).
        torch.testing.assert_close(x_n.cpu(), x)

    @pytest.mark.parametrize(
        "op,args",
        [
            (aten.upsample_nearest2d, ([7, 9], None, None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None, None)),
        ],
    )
    def test_out_overload_non_contiguous(self, op, args):
        """A strided ``out=`` must still receive the right values."""
        x = torch.randn(2, 3, 4, 5)
        expected = op(x, *args)
        big = torch.empty(2, 3, 7, 18, device="neuron", dtype=expected.dtype)
        out_n = big[..., ::2]
        assert not out_n.is_contiguous()
        with track_neuron_ops():
            op.out(x.to("neuron"), *args, out=out_n)
            _assert_ran_on_device(op)
        torch.testing.assert_close(out_n.cpu(), expected)

    @pytest.mark.parametrize(
        "op,size_args",
        [
            (aten.upsample_nearest1d, ([8], None)),
            (aten._upsample_nearest_exact1d, ([8], None)),
            (aten.upsample_nearest2d, ([7, 9], None)),
            (aten._upsample_nearest_exact2d, ([7, 9], None)),
        ],
    )
    def test_vec_overload_output_size(self, op, size_args):
        rank = 1 if "1d" in str(op) else 2
        x = torch.randn(2, 3, 5) if rank == 1 else torch.randn(2, 3, 4, 5)
        expected = op.vec(x, *size_args)
        with track_neuron_ops():
            got = op.vec(x.to("neuron"), *size_args)
            _assert_ran_on_device(op)
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize(
        "op,vec_args",
        [
            (aten.upsample_nearest1d, (None, [2.0])),
            (aten.upsample_nearest2d, (None, [2.0, 3.0])),
            (aten.upsample_linear1d, (None, False, [2.0])),
            (aten.upsample_bilinear2d, (None, False, [2.0, 3.0])),
            (aten.upsample_bicubic2d, (None, False, [2.0, 3.0])),
        ],
    )
    def test_vec_overload_scale_factors(self, op, vec_args):
        rank = 1 if "1d" in str(op) else 2
        x = torch.randn(2, 3, 5) if rank == 1 else torch.randn(2, 3, 4, 5)
        expected = op.vec(x, *vec_args)
        with track_neuron_ops():
            got = op.vec(x.to("neuron"), *vec_args)
            _assert_ran_on_device(op)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize(
        "op,vec_args",
        [
            (aten.upsample_nearest2d, ([7, 9], None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None)),
        ],
    )
    def test_vec_out_overload(self, op, vec_args):
        """``vec_out`` exists only for nearest2d and bilinear2d."""
        x = torch.randn(2, 3, 4, 5)
        expected = op.vec(x, *vec_args)
        out_n = torch.empty_like(expected).to("neuron")
        with track_neuron_ops():
            result = op.vec_out(x.to("neuron"), *vec_args, out=out_n)
            _assert_ran_on_device(op)
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(out_n.cpu(), expected)


# ═════════════════════════ edge cases and errors ═════════════════════════
class TestEdgeCases:
    @pytest.mark.parametrize(
        "op,shape,args",
        [
            (aten.upsample_nearest1d, (0, 3, 4), ([8], None)),
            (aten.upsample_nearest2d, (0, 3, 4, 5), ([7, 9], None, None)),
            (aten.upsample_bilinear2d, (0, 3, 4, 5), ([7, 9], False, None, None)),
            (aten.upsample_bicubic2d, (0, 3, 4, 5), ([7, 9], False, None, None)),
            (aten.upsample_linear1d, (0, 3, 4), ([8], False, None)),
        ],
    )
    def test_zero_batch(self, op, shape, args):
        x = torch.randn(*shape)
        expected = op(x, *args)
        with track_neuron_ops():
            got = op(x.to("neuron"), *args)
        assert got.shape == expected.shape
        assert got.numel() == 0

    @pytest.mark.parametrize(
        "op,shape,args",
        [
            (aten.upsample_nearest1d, (1, 1, 1), ([1], None)),
            (aten.upsample_nearest2d, (1, 1, 1, 1), ([1, 1], None, None)),
            (aten.upsample_bilinear2d, (1, 1, 1, 1), ([1, 1], False, None, None)),
            (aten.upsample_bicubic2d, (1, 1, 1, 1), ([3, 3], False, None, None)),
            (aten.upsample_linear1d, (1, 1, 1), ([5], False, None)),
        ],
    )
    def test_single_element(self, op, shape, args):
        x = torch.randn(*shape)
        expected = op(x, *args)
        with track_neuron_ops():
            got = op(x.to("neuron"), *args)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
    @pytest.mark.parametrize(
        "op,shape,args",
        [
            (aten.upsample_nearest1d, (1, 1, 4), ([8], None)),
            (aten.upsample_nearest2d, (1, 1, 4, 4), ([8, 8], None, None)),
            (aten.upsample_bilinear2d, (1, 1, 4, 4), ([8, 8], False, None, None)),
            (aten.upsample_bicubic2d, (1, 1, 4, 4), ([8, 8], False, None, None)),
            (aten.upsample_linear1d, (1, 1, 4), ([8], False, None)),
        ],
    )
    def test_rejects_non_float_non_uint8_dtypes(self, dtype, op, shape, args):
        """aten's kernels dispatch over floating + uint8 only; we must match."""
        x = torch.ones(*shape, dtype=dtype, device="neuron")
        with pytest.raises((RuntimeError, NotImplementedError), match="not implemented for"):
            op(x, *args)

    def test_linear1d_rejects_uint8(self):
        """``upsample_linear1d`` is floating-only even though bilinear2d takes uint8."""
        x = torch.ones(1, 1, 4, dtype=torch.uint8, device="neuron")
        with pytest.raises((RuntimeError, NotImplementedError), match="not implemented for"):
            aten.upsample_linear1d(x, [8], False)
        # ... and the CPU reference agrees, so this is not an invented rule.
        with pytest.raises((RuntimeError, NotImplementedError), match="not implemented for"):
            aten.upsample_linear1d(torch.ones(1, 1, 4, dtype=torch.uint8), [8], False)

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.int64])
    @pytest.mark.parametrize(
        "op,shape,args",
        [
            (aten.upsample_nearest1d_backward, (1, 1, 8), ([8], [1, 1, 4], None)),
            (aten.upsample_nearest2d_backward, (1, 1, 8, 8), ([8, 8], [1, 1, 4, 4], None, None)),
            (
                aten.upsample_bilinear2d_backward,
                (1, 1, 8, 8),
                ([8, 8], [1, 1, 4, 4], False, None, None),
            ),
            (
                aten.upsample_bicubic2d_backward,
                (1, 1, 8, 8),
                ([8, 8], [1, 1, 4, 4], False, None, None),
            ),
            (aten.upsample_linear1d_backward, (1, 1, 8), ([8], [1, 1, 4], False, None)),
        ],
    )
    def test_backwards_reject_integral_dtypes(self, dtype, op, shape, args):
        """Every backward is floating-only on CPU; the guards must mirror that."""
        g = torch.ones(*shape, dtype=dtype, device="neuron")
        with pytest.raises((RuntimeError, NotImplementedError), match="not implemented for"):
            op(g, *args)


# ═════════════════════════ memory format ═════════════════════════
class TestMemoryFormat:
    """A channels_last input must produce a channels_last output.

    Eager kernels allocate contiguous output buffers by default, so the ops
    request a layout via ``register_aten(lowering_result_transform=...)`` — the
    output-layout metadata path convolution and batch norm use. The strides come
    from aten's meta kernel; OutputProcessing then allocates in stride order,
    letting the epilogue fuse the relayout into the NEFF.

    aten's own answer is the reference, including where it is *contiguous* for a
    channels_last input (a 3-D unsqueezed batch-1 case), so these compare strides
    against the CPU op rather than asserting channels_last unconditionally.
    """

    CL = torch.channels_last

    @pytest.mark.parametrize(
        "op,args",
        [
            (aten.upsample_nearest2d, ([7, 9], None, None)),
            (aten._upsample_nearest_exact2d, ([7, 9], None, None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None, None)),
            (aten.upsample_bilinear2d, ([7, 9], True, None, None)),
            (aten.upsample_bicubic2d, ([7, 9], False, None, None)),
            (aten.upsample_bicubic2d, ([7, 9], True, None, None)),
        ],
    )
    @pytest.mark.parametrize("batch", [1, 5])
    def test_channels_last_preserved(self, op, args, batch):
        x = torch.randn(batch, 3, 4, 5).contiguous(memory_format=self.CL)
        expected = op(x, *args)
        with track_neuron_ops():
            got = op(x.to("neuron"), *args)
            _assert_ran_on_device(op)
        assert (
            got.stride() == expected.stride()
        ), f"stride {got.stride()} != aten's {expected.stride()}"
        assert got.is_contiguous(memory_format=self.CL) == expected.is_contiguous(
            memory_format=self.CL
        )
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize(
        "op,args",
        [
            (aten.upsample_nearest2d, ([7, 9], None, None)),
            (aten._upsample_nearest_exact2d, ([7, 9], None, None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None, None)),
            (aten.upsample_bicubic2d, ([7, 9], True, None, None)),
        ],
    )
    def test_channels_last_survives_float64_bounce(self, op, args):
        """float64 has no device dtype, so the op runs in fp32 and the result is
        upcast back through a CPU round-trip that returns a *contiguous* tensor.
        ``ApplyTargetStrides`` at the tail of UpcastExecutionOutputs re-imposes the
        requested (channels_last) strides on the fp64 result; without it the layout
        is silently dropped.
        """
        x = torch.randn(5, 3, 4, 5, dtype=torch.double).contiguous(memory_format=self.CL)
        expected = op(x, *args)
        with track_neuron_ops():
            got = op(x.to("neuron"), *args)
            _assert_ran_on_device(op)
        assert got.dtype == torch.double
        assert (
            got.stride() == expected.stride()
        ), f"stride {got.stride()} != aten's {expected.stride()}"
        assert got.is_contiguous(memory_format=self.CL) == expected.is_contiguous(
            memory_format=self.CL
        )
        # fp32 tolerances: Neuron has no float64 so the op runs in fp32; this test
        # guards the *layout*, and float64 precision is covered elsewhere.
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "op,args",
        [
            (aten.upsample_nearest2d, ([7, 9], None, None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None, None)),
            (aten.upsample_bicubic2d, ([7, 9], True, None, None)),
        ],
    )
    def test_layout_hook_ignores_the_out_tensor(self, op, args):
        """The hook must give the same answer however ``out=`` is passed.

        The structured overload passes ``out`` **positionally**, and all the
        ``F.interpolate`` traffic funnels into it. Forwarding that tensor to the
        aten meta call raises, so the transform must filter it out — otherwise the
        layout is silently disabled for the only overload that matters, while the
        functional form (what a unit test naturally writes) keeps working.
        """
        from torch_neuronx.python_ops.torch_mlir.ops.interpolation import (
            _make_output_stride_transform,
        )

        transform = _make_output_stride_transform(op)
        x = torch.randn(2, 3, 4, 5).contiguous(memory_format=self.CL)
        out = torch.empty(2, 3, 7, 9)
        out_shape = tuple(op(x, *args).shape)

        def target_strides(*call_args, **call_kwargs):
            """Run the transform on a stub LoweringResult and read back the strides."""
            result = SimpleNamespace(output_specs=[(out_shape, torch.float32)])
            transform(result, (x, *call_args), call_kwargs)
            spec = result.output_specs[0]
            return spec[2] if len(spec) > 2 else None

        functional = target_strides(*args)
        assert functional is not None, "expected a layout for a channels_last input"
        assert target_strides(*args, out=out) == functional
        assert target_strides(*args, out) == functional

    def test_out_overload_preserves_channels_last(self):
        """End-to-end version of the above, through the .out overload on device."""
        x = torch.randn(2, 3, 4, 5).contiguous(memory_format=self.CL)
        expected = aten.upsample_bilinear2d(x, [7, 9], False)
        out_n = torch.empty_strided(
            expected.shape, expected.stride(), device="neuron", dtype=expected.dtype
        )
        with track_neuron_ops():
            aten.upsample_bilinear2d.out(x.to("neuron"), [7, 9], False, out=out_n)
        assert out_n.stride() == expected.stride()
        torch.testing.assert_close(out_n.cpu(), expected)

    @pytest.mark.parametrize(
        "op,args",
        [
            (aten.upsample_nearest2d, ([7, 9], None, None)),
            (aten.upsample_bilinear2d, ([7, 9], False, None, None)),
        ],
    )
    def test_contiguous_input_stays_contiguous(self, op, args):
        """The layout hook must not perturb the ordinary contiguous path."""
        x = torch.randn(2, 3, 4, 5)
        expected = op(x, *args)
        with track_neuron_ops():
            got = op(x.to("neuron"), *args)
        assert got.is_contiguous()
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got.cpu(), expected)


# ═════════════════════════ nn.functional entry points ═════════════════════════
class TestFunctionalInterface:
    """``F.interpolate`` / ``nn.Upsample`` route through the ``.vec`` overloads."""

    @pytest.mark.parametrize(
        "mode,kwargs",
        [
            ("nearest", {}),
            ("nearest-exact", {}),
            ("linear", {"align_corners": False}),
            ("linear", {"align_corners": True}),
        ],
    )
    def test_interpolate_1d(self, mode, kwargs):
        x = torch.randn(2, 3, 5)
        expected = torch.nn.functional.interpolate(x, size=8, mode=mode, **kwargs)
        with track_neuron_ops():
            got = torch.nn.functional.interpolate(x.to("neuron"), size=8, mode=mode, **kwargs)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize(
        "mode,kwargs",
        [
            ("nearest", {}),
            ("nearest-exact", {}),
            ("bilinear", {"align_corners": False}),
            ("bilinear", {"align_corners": True}),
            ("bicubic", {"align_corners": False}),
            ("bicubic", {"align_corners": True}),
        ],
    )
    def test_interpolate_2d(self, mode, kwargs):
        x = torch.randn(2, 3, 4, 5)
        expected = torch.nn.functional.interpolate(x, size=(7, 9), mode=mode, **kwargs)
        with track_neuron_ops():
            got = torch.nn.functional.interpolate(x.to("neuron"), size=(7, 9), mode=mode, **kwargs)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
    def test_interpolate_scale_factor(self, mode):
        """scale_factor routes through ``.vec`` with ``output_size=None``."""
        x = torch.randn(2, 3, 4, 5)
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        expected = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=mode, **kwargs)
        with track_neuron_ops():
            got = torch.nn.functional.interpolate(
                x.to("neuron"), scale_factor=2.0, mode=mode, **kwargs
            )
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize(
        "module",
        [
            torch.nn.Upsample(size=(7, 9), mode="nearest"),
            torch.nn.Upsample(scale_factor=2, mode="nearest"),
            torch.nn.Upsample(size=(7, 9), mode="bilinear", align_corners=False),
            torch.nn.UpsamplingNearest2d(size=(7, 9)),
            torch.nn.UpsamplingBilinear2d(size=(7, 9)),
        ],
    )
    def test_upsample_modules(self, module):
        x = torch.randn(2, 3, 4, 5)
        expected = module(x)
        with track_neuron_ops():
            got = module(x.to("neuron"))
        torch.testing.assert_close(got.cpu(), expected)


# ═════════════════════════ anti-aliased bilinear ═════════════════════════
# aten's CPU AA kernel raises for fp16/bf16, so for those dtypes the reference
# is the fp32 CPU result, not a same-dtype CPU run. This is the crash the Neuron
# path fixes: the customer models could not run the op on CPU in bf16 at all.

# ((in_h, in_w), (out_h, out_w), (scales_h, scales_w))
AA_2D_CFGS = [
    ((8, 8), (4, 4), (None, None)),  # 2x downsample, antialias active
    ((10, 7), (5, 4), (None, None)),  # non-square downsample
    ((9, 9), (3, 3), (None, None)),  # 3x downsample
    ((8, 8), (8, 8), (None, None)),  # identity
    ((4, 4), (8, 8), (None, None)),  # upsample
]


class TestAntiAliasedBilinearForward:
    """aten::_upsample_bilinear2d_aa forward on Neuron."""

    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_hw,out_hw,scales", AA_2D_CFGS)
    def test_fp32_values(self, align_corners, in_hw, out_hw, scales):
        x = torch.randn(2, 3, *in_hw)
        expected = aten._upsample_bilinear2d_aa(x, list(out_hw), align_corners, *scales)
        with track_neuron_ops():
            got = aten._upsample_bilinear2d_aa(x.to("neuron"), list(out_hw), align_corners, *scales)
            _assert_ran_on_device(aten._upsample_bilinear2d_aa)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("in_hw,out_hw,scales", AA_2D_CFGS)
    def test_reduced_precision_matches_fp32_oracle(self, dtype, in_hw, out_hw, scales):
        """fp16/bf16 run on Neuron but crash on CPU, so compare to the fp32 CPU oracle."""
        x = torch.randn(2, 3, *in_hw)
        if tuple(in_hw) != tuple(out_hw):
            # A real resample crashes on CPU for narrow dtypes. aten short-circuits
            # out == in to a copy that never reaches the unimplemented kernel.
            with pytest.raises(NotImplementedError, match="not implemented"):
                aten._upsample_bilinear2d_aa(x.to(dtype), list(out_hw), False, *scales)
        oracle = aten._upsample_bilinear2d_aa(x, list(out_hw), False, *scales)
        with track_neuron_ops():
            got = aten._upsample_bilinear2d_aa(
                x.to(dtype).to("neuron"), list(out_hw), False, *scales
            )
            _assert_ran_on_device(aten._upsample_bilinear2d_aa)
        assert got.dtype == dtype
        tol = (
            {"atol": 5e-2, "rtol": 5e-2}
            if dtype == torch.bfloat16
            else {"atol": 1e-2, "rtol": 1e-2}
        )
        torch.testing.assert_close(got.cpu().float(), oracle, **tol)

    @pytest.mark.parametrize("in_hw,out_hw", [((8, 8), (4, 4)), ((7, 5), (3, 4))])
    def test_uint8_matches_cpu(self, in_hw, out_hw):
        x = torch.randint(0, 256, (2, 3, *in_hw), dtype=torch.uint8)
        expected = aten._upsample_bilinear2d_aa(x, list(out_hw), False)
        with track_neuron_ops():
            got = aten._upsample_bilinear2d_aa(x.to("neuron"), list(out_hw), False)
            _assert_ran_on_device(aten._upsample_bilinear2d_aa)
        assert got.dtype == torch.uint8
        diff = (got.cpu().to(torch.int16) - expected.to(torch.int16)).abs()
        assert diff.max().item() <= 1


class TestAntiAliasedBilinearBackward:
    """aten::_upsample_bilinear2d_aa_backward on Neuron. Floating point only."""

    @pytest.mark.parametrize("align_corners", [False, True])
    @pytest.mark.parametrize("in_hw,out_hw,scales", AA_2D_CFGS)
    def test_fp32_values(self, align_corners, in_hw, out_hw, scales):
        op = aten._upsample_bilinear2d_aa_backward
        grad = torch.randn(2, 3, *out_hw)
        input_size = [2, 3, *in_hw]
        expected = op(grad, list(out_hw), input_size, align_corners, *scales)
        with track_neuron_ops():
            got = op(grad.to("neuron"), list(out_hw), input_size, align_corners, *scales)
            _assert_ran_on_device(op)
        torch.testing.assert_close(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("in_hw,out_hw,scales", AA_2D_CFGS)
    def test_reduced_precision_matches_fp32_oracle(self, dtype, in_hw, out_hw, scales):
        op = aten._upsample_bilinear2d_aa_backward
        grad = torch.randn(2, 3, *out_hw)
        input_size = [2, 3, *in_hw]
        oracle = op(grad, list(out_hw), input_size, False, *scales)
        with track_neuron_ops():
            got = op(grad.to(dtype).to("neuron"), list(out_hw), input_size, False, *scales)
            _assert_ran_on_device(op)
        assert got.dtype == dtype
        tol = (
            {"atol": 5e-2, "rtol": 5e-2}
            if dtype == torch.bfloat16
            else {"atol": 1e-2, "rtol": 1e-2}
        )
        torch.testing.assert_close(got.cpu().float(), oracle, **tol)
