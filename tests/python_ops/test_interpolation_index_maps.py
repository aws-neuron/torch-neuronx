"""CPU-only sweep pinning the upsample index/weight maps against aten.

Every op in the upsample family is a *linear* map of its input for a fixed
(in_size, out_size, scale, align_corners) configuration, so the exact operator
each aten kernel implements can be recovered by pushing an identity basis
through the CPU kernel. These tests diff the trace-time maps built by
``ops/interpolation.py`` against that operator over an ``itertools.product``
sweep, in float64.

No Neuron device is needed, so this runs everywhere and it is where the aten
asymmetries are actually pinned — a few thousand comparisons in seconds. The
device tests in ``test_interpolation.py`` then only have to confirm that the
lowering applies these maps correctly on hardware.
"""

import itertools

import pytest
import torch

from torch_neuronx._interpolation_maps import (
    bicubic_taps as _bicubic_taps,
)
from torch_neuronx._interpolation_maps import (
    linear_taps as _linear_taps,
)
from torch_neuronx._interpolation_maps import (
    nearest_indices as _nearest_indices,
)

aten = torch.ops.aten

# float32, not float64. aten computes the source index and the tap weights in the
# input's opmath type, so a float64 basis would make aten resolve indices in double
# precision while the lowering — which has to run on a device with no float64 —
# resolves them in float32. Comparing against the float64 behaviour moved taps at
# boundary ratios and made this sweep reject correct maps.
DT = torch.float32
# A few float32 ULP: bicubic sums four taps and the 2-D check krons two of those
# together, so a handful of roundings accumulate. Still tight enough that a moved
# tap — the failure mode this sweep exists to catch — is orders of magnitude larger.
TAP_ATOL = 5e-6

SIZES = [1, 2, 3, 4, 5, 8]
SCALES = [None, 0.5, 1.0, 1.5, 2.0, 3.0]


def operator_of(fn, in_spatial):
    """The (prod(out_spatial), prod(in_spatial)) matrix ``fn`` applies."""
    n_in = 1
    for d in in_spatial:
        n_in *= d
    basis = torch.eye(n_in, dtype=DT).reshape(n_in, 1, *in_spatial)
    out = fn(basis)
    return out.reshape(n_in, -1).T.contiguous()


def nearest_matrix(in_size, out_size, scale, exact, short_circuit):
    """The operator implied by the map builder, evaluated on CPU.

    The builders are device-agnostic (they are built from ``arange``, so they work
    below AOTAutograd), which means this sweep exercises the very same code the
    device runs rather than a parallel reference implementation.
    """
    indices = _nearest_indices(
        in_size, out_size, scale, exact=exact, short_circuit=short_circuit, device="cpu"
    )
    m = torch.zeros(out_size, in_size, dtype=DT)
    m[torch.arange(out_size), indices.long()] = 1.0
    return m


def taps_matrix(in_size, out_size, scale, align_corners, cubic):
    builder = _bicubic_taps if cubic else _linear_taps
    indices, weights = builder(in_size, out_size, scale, align_corners, "cpu")
    m = torch.zeros(out_size, in_size, dtype=DT)
    rows = torch.arange(out_size)
    for tap_indices, tap_weights in zip(indices, weights, strict=True):
        m[rows, tap_indices.long()] += tap_weights.to(DT)
    return m


# ── nearest ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("in_w,out_w", list(itertools.product(SIZES, SIZES)))
@pytest.mark.parametrize("scale", SCALES)
def test_nearest_1d_map(exact, in_w, out_w, scale):
    """The 1-D forward takes no fast path, so an explicit scale always applies."""
    op = aten._upsample_nearest_exact1d if exact else aten.upsample_nearest1d
    got = operator_of(lambda x: op(x, [out_w], scale), (in_w,))
    want = nearest_matrix(in_w, out_w, scale, exact, short_circuit=False)
    assert torch.equal(got, want)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize(
    "in_hw,out_hw",
    list(
        itertools.product(
            itertools.product([1, 3, 4], [1, 3, 4]), itertools.product([1, 3, 4, 6], [1, 3, 4, 6])
        )
    ),
)
@pytest.mark.parametrize("scales", [(None, None), (2.0, 0.5), (3.0, None)])
def test_nearest_2d_map(exact, in_hw, out_hw, scales):
    """The 2-D forward short-circuits out == in and out == 2*in, ignoring scales."""
    op = aten._upsample_nearest_exact2d if exact else aten.upsample_nearest2d
    got = operator_of(lambda x: op(x, list(out_hw), *scales), in_hw)
    want = torch.kron(
        nearest_matrix(in_hw[0], out_hw[0], scales[0], exact, short_circuit=not exact),
        nearest_matrix(in_hw[1], out_hw[1], scales[1], exact, short_circuit=not exact),
    )
    assert torch.equal(got, want)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("in_w,out_w", [(3, 3), (4, 4), (3, 6), (4, 8), (4, 5)])
@pytest.mark.parametrize("scale", [None, 0.5, 2.0])
def test_nearest_1d_backward_uses_short_circuit_map(exact, in_w, out_w, scale):
    """Both nearest backwards use the short-circuiting map, at both ranks.

    For the plain (non-exact) 1-D case that makes the backward *not* the adjoint
    of the forward — the asymmetry this test exists to pin.
    """
    op = aten._upsample_nearest_exact1d_backward if exact else aten.upsample_nearest1d_backward
    got = operator_of(lambda g: op(g, [out_w], [g.shape[0], 1, in_w], scale), (out_w,))
    want = nearest_matrix(in_w, out_w, scale, exact, short_circuit=not exact)
    assert torch.equal(got, want.T)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("in_hw,out_hw", [((3, 3), (3, 3)), ((2, 3), (4, 6)), ((3, 2), (4, 5))])
@pytest.mark.parametrize("scales", [(None, None), (2.0, 0.5)])
def test_nearest_2d_backward_map(exact, in_hw, out_hw, scales):
    op = aten._upsample_nearest_exact2d_backward if exact else aten.upsample_nearest2d_backward
    got = operator_of(lambda g: op(g, list(out_hw), [g.shape[0], 1, *in_hw], *scales), out_hw)
    want = torch.kron(
        nearest_matrix(in_hw[0], out_hw[0], scales[0], exact, short_circuit=not exact),
        nearest_matrix(in_hw[1], out_hw[1], scales[1], exact, short_circuit=not exact),
    )
    assert torch.equal(got, want.T)


# ── linear / bilinear / bicubic ──────────────────────────────────────────────
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("in_w,out_w", list(itertools.product(SIZES, SIZES)))
@pytest.mark.parametrize("scale", SCALES)
def test_linear1d_map(align_corners, in_w, out_w, scale):
    got = operator_of(lambda x: aten.upsample_linear1d(x, [out_w], align_corners, scale), (in_w,))
    want = taps_matrix(in_w, out_w, scale, align_corners, cubic=False)
    torch.testing.assert_close(got, want, atol=TAP_ATOL, rtol=0)


@pytest.mark.parametrize("cubic", [False, True])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "in_hw,out_hw",
    list(
        itertools.product(
            itertools.product([2, 3, 5], [2, 3, 5]),
            itertools.product([1, 2, 3, 5, 7], [1, 2, 3, 5, 7]),
        )
    ),
)
@pytest.mark.parametrize("scales", [(None, None), (2.0, 0.5)])
def test_bimode2d_map(cubic, align_corners, in_hw, out_hw, scales):
    """2-D is separable, and the linear family short-circuits per dimension."""
    op = aten.upsample_bicubic2d if cubic else aten.upsample_bilinear2d
    got = operator_of(lambda x: op(x, list(out_hw), align_corners, *scales), in_hw)
    want = torch.kron(
        taps_matrix(in_hw[0], out_hw[0], scales[0], align_corners, cubic),
        taps_matrix(in_hw[1], out_hw[1], scales[1], align_corners, cubic),
    )
    torch.testing.assert_close(got, want, atol=TAP_ATOL, rtol=0)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("in_w,out_w", [(1, 4), (2, 5), (3, 3), (5, 2), (4, 7)])
@pytest.mark.parametrize("scale", [None, 0.5, 2.0])
def test_linear1d_backward_is_adjoint(align_corners, in_w, out_w, scale):
    got = operator_of(
        lambda g: aten.upsample_linear1d_backward(
            g, [out_w], [g.shape[0], 1, in_w], align_corners, scale
        ),
        (out_w,),
    )
    want = taps_matrix(in_w, out_w, scale, align_corners, cubic=False)
    torch.testing.assert_close(got, want.T, atol=TAP_ATOL, rtol=0)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("in_hw,out_hw", [((2, 3), (4, 6)), ((3, 2), (5, 7)), ((3, 3), (2, 2))])
@pytest.mark.parametrize("scales", [(None, None), (2.0, 0.5)])
def test_bilinear2d_backward_is_adjoint(align_corners, in_hw, out_hw, scales):
    got = operator_of(
        lambda g: aten.upsample_bilinear2d_backward(
            g, list(out_hw), [g.shape[0], 1, *in_hw], align_corners, *scales
        ),
        out_hw,
    )
    want = torch.kron(
        taps_matrix(in_hw[0], out_hw[0], scales[0], align_corners, cubic=False),
        taps_matrix(in_hw[1], out_hw[1], scales[1], align_corners, cubic=False),
    )
    torch.testing.assert_close(got, want.T, atol=TAP_ATOL, rtol=0)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("in_hw,out_hw", [((2, 3), (4, 6)), ((3, 2), (5, 7)), ((3, 3), (2, 2))])
@pytest.mark.parametrize("scales", [(None, None), (2.0, 0.5)])
def test_bicubic2d_backward_is_adjoint_when_sizes_change(align_corners, in_hw, out_hw, scales):
    got = operator_of(
        lambda g: aten.upsample_bicubic2d_backward(
            g, list(out_hw), [g.shape[0], 1, *in_hw], align_corners, *scales
        ),
        out_hw,
    )
    want = torch.kron(
        taps_matrix(in_hw[0], out_hw[0], scales[0], align_corners, cubic=True),
        taps_matrix(in_hw[1], out_hw[1], scales[1], align_corners, cubic=True),
    )
    torch.testing.assert_close(got, want.T, atol=TAP_ATOL, rtol=0)


@pytest.mark.parametrize("in_hw", [(2, 2), (2, 3), (3, 3)])
@pytest.mark.parametrize("scales", [(None, 0.5), (2.0, None), (2.0, 0.5)])
def test_bicubic2d_backward_copies_when_both_sizes_unchanged(in_hw, scales):
    """Quirk 3: the copy short-circuit ignores the scales its forward honours.

    Only observable with an explicit scale — without one the adjoint happens to
    be the identity anyway.
    """
    got = operator_of(
        lambda g: aten.upsample_bicubic2d_backward(
            g, list(in_hw), [g.shape[0], 1, *in_hw], False, *scales
        ),
        in_hw,
    )
    assert torch.equal(got, torch.eye(in_hw[0] * in_hw[1], dtype=DT))
    # The forward with the same scales is NOT the identity, so the asymmetry is real.
    forward = operator_of(lambda x: aten.upsample_bicubic2d(x, list(in_hw), False, *scales), in_hw)
    assert not torch.equal(forward, torch.eye(in_hw[0] * in_hw[1], dtype=DT))
