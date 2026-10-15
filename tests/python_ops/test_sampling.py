"""Eager tests for the aten grid_sampler family on Neuron.

Covers the forward and backward of ``grid_sampler_2d`` lowered in
``python_ops/torch_mlir/ops/sampling.py``. CPU is the reference throughout, so
the tests track upstream behaviour rather than hand-written constants.

Interpolation modes: 0 = bilinear, 1 = nearest, 2 = bicubic.
Padding modes:       0 = zeros, 1 = border, 2 = reflection.
"""

import itertools

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

aten = torch.ops.aten

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

MODES_2D = [0, 1, 2]
PADDING_MODES = [0, 1, 2]
ALIGN = [False, True]

# (input NCHW, output (H_out, W_out)); grid is (N, H_out, W_out, 2)
CFGS_2D = [
    ((2, 3, 5, 6), (7, 8)),
    ((1, 1, 4, 4), (4, 4)),  # single channel/batch
    ((2, 4, 1, 7), (3, 5)),  # size-1 spatial dim
    ((1, 2, 8, 3), (1, 1)),  # single output point
]


def _tol(dtype):
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    slack = 16 * torch.finfo(dtype).eps
    return {"atol": slack, "rtol": slack}


def _grid(shape, dtype):
    # Range [-2, 2] exercises both in-range and out-of-range samples.
    return (torch.rand(*shape, dtype=torch.float32) * 4 - 2).to(dtype)


def _nearest_safe_grid(shape, in_shape, align_corners, dtype):
    """Return a grid whose nearest-mode source coordinates hit pixel centers.

    Nearest sampling is a discrete pixel pick. A coordinate within rounding
    epsilon of a half-integer boundary can round to a different pixel on Neuron
    than on the fp32 CPU reference. The two outputs then differ by a whole pixel
    value, which no relative tolerance covers. Snap each axis to a pixel center
    to remove the tie. Bilinear and bicubic are continuous and do not need this.
    """
    g = torch.rand(*shape, dtype=torch.float32) * 4 - 2
    height, width = in_shape[-2], in_shape[-1]

    def snap(coord, size):
        if align_corners:
            if size == 1:
                return torch.zeros_like(coord)
            src = ((coord + 1) / 2 * (size - 1)).round()
            return 2 * src / (size - 1) - 1
        src = ((coord + 1) / 2 * size - 0.5).round().clamp(0, size - 1)
        return (2 * src + 1) / size - 1

    g[..., 0] = snap(g[..., 0], width)
    g[..., 1] = snap(g[..., 1], height)
    return g.to(dtype)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    # Seed every test so the random inputs are reproducible. An unseeded
    # nearest-mode test can flip a pixel pick at a boundary and flake.
    torch.manual_seed(0)


# --- forward ---
class TestForward:
    @pytest.mark.parametrize(
        "mode,padding,align", list(itertools.product(MODES_2D, PADDING_MODES, ALIGN))
    )
    @pytest.mark.parametrize("in_shape,grid_hw", CFGS_2D)
    def test_2d_values(self, mode, padding, align, in_shape, grid_hw):
        x = torch.randn(*in_shape)
        grid_shape = (in_shape[0], *grid_hw, 2)
        if mode == 1:
            g = _nearest_safe_grid(grid_shape, in_shape, align, torch.float32)
        else:
            g = _grid(grid_shape, torch.float32)
        expected = aten.grid_sampler_2d(x, g, mode, padding, align)
        with track_neuron_ops():
            got = aten.grid_sampler_2d(x.to("neuron"), g.to("neuron"), mode, padding, align)
            assert_op_runs_on_neuron("aten::grid_sampler_2d")
        assert got.is_contiguous()
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("mode", MODES_2D)
    def test_2d_dtypes(self, dtype, mode):
        in_shape = (2, 3, 6, 7)
        x = torch.randn(*in_shape, dtype=dtype)
        if mode == 1:
            g = _nearest_safe_grid((2, 5, 5, 2), in_shape, False, dtype)
        else:
            g = _grid((2, 5, 5, 2), dtype)
        expected = aten.grid_sampler_2d(x, g, mode, 0, False)
        with track_neuron_ops():
            got = aten.grid_sampler_2d(x.to("neuron"), g.to("neuron"), mode, 0, False)
            assert_op_runs_on_neuron("aten::grid_sampler_2d")
        assert got.dtype == dtype
        torch.testing.assert_close(got.cpu(), expected, **_tol(dtype))

    def test_2d_channels_last_matches_cpu(self):
        # aten returns a contiguous output even for a channels_last input, so the
        # Neuron output must match both the values and the contiguous layout.
        x = torch.randn(2, 3, 6, 7).to(memory_format=torch.channels_last)
        g = _grid((2, 5, 5, 2), torch.float32)
        expected = aten.grid_sampler_2d(x, g, 0, 0, False)
        with track_neuron_ops():
            got = aten.grid_sampler_2d(x.to("neuron"), g.to("neuron"), 0, 0, False)
            assert_op_runs_on_neuron("aten::grid_sampler_2d")
        assert got.is_contiguous()
        assert got.is_contiguous() == expected.is_contiguous()
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)

    def test_2d_out_writes_value_and_leaves_self_untouched(self):
        x = torch.randn(2, 3, 6, 7)
        g = _grid((2, 5, 5, 2), torch.float32)
        expected = aten.grid_sampler_2d(x, g, 0, 0, False)
        xn = x.to("neuron")
        gn = g.to("neuron")
        out = torch.empty(2, 3, 5, 5, device="neuron")
        with track_neuron_ops():
            ret = aten.grid_sampler_2d.out(xn, gn, 0, 0, False, out=out)
            assert_op_runs_on_neuron("aten::grid_sampler_2d")
        torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ret.cpu(), expected, atol=1e-4, rtol=1e-4)
        # self must be unchanged by the out= write.
        torch.testing.assert_close(xn.cpu(), x)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_non_float_raises(self, dtype):
        x = torch.ones(2, 3, 5, 5, dtype=dtype).to("neuron")
        g = torch.zeros(2, 4, 4, 2).to("neuron")
        with pytest.raises((NotImplementedError, RuntimeError), match="grid_sampler_2d"):
            aten.grid_sampler_2d(x, g, 0, 0, False)


# --- backward op ---
class TestBackwardOp:
    @pytest.mark.parametrize(
        "mode,padding,align", list(itertools.product(MODES_2D, PADDING_MODES, ALIGN))
    )
    def test_2d_backward_values(self, mode, padding, align):
        x = torch.randn(2, 3, 5, 6)
        g = _grid((2, 7, 8, 2), torch.float32)
        go = torch.randn(2, 3, 7, 8)
        ei, eg = aten.grid_sampler_2d_backward(go, x, g, mode, padding, align, [True, True])
        with track_neuron_ops():
            gi, gg = aten.grid_sampler_2d_backward(
                go.to("neuron"), x.to("neuron"), g.to("neuron"), mode, padding, align, [True, True]
            )
            assert_op_runs_on_neuron("aten::grid_sampler_2d_backward")
        torch.testing.assert_close(gi.cpu(), ei, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gg.cpu(), eg, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("output_mask", [[True, False], [False, True]])
    def test_2d_backward_output_mask(self, output_mask):
        x = torch.randn(2, 3, 5, 6)
        g = _grid((2, 7, 8, 2), torch.float32)
        go = torch.randn(2, 3, 7, 8)
        ei, eg = aten.grid_sampler_2d_backward(go, x, g, 0, 0, False, output_mask)
        with track_neuron_ops():
            gi, gg = aten.grid_sampler_2d_backward(
                go.to("neuron"), x.to("neuron"), g.to("neuron"), 0, 0, False, output_mask
            )
            assert_op_runs_on_neuron("aten::grid_sampler_2d_backward")
        if output_mask[0]:
            torch.testing.assert_close(gi.cpu(), ei, atol=1e-3, rtol=1e-3)
        if output_mask[1]:
            torch.testing.assert_close(gg.cpu(), eg, atol=1e-3, rtol=1e-3)


# --- autograd end-to-end ---
class TestAutograd:
    @pytest.mark.parametrize("mode", MODES_2D)
    @pytest.mark.parametrize("padding", PADDING_MODES)
    def test_2d_autograd(self, mode, padding):
        x = torch.randn(2, 3, 5, 6)
        g = _grid((2, 7, 8, 2), torch.float32)
        xn = x.clone().to("neuron").requires_grad_()
        gn = g.clone().to("neuron").requires_grad_()
        with track_neuron_ops():
            out = aten.grid_sampler_2d(xn, gn, mode, padding, False)
            out.sum().backward()
            # The forward records grid_sampler_2d, whose backward dispatches the
            # separate grid_sampler_2d_backward op; assert IT ran on Neuron.
            assert_op_runs_on_neuron("aten::grid_sampler_2d_backward")
        xc = x.clone().requires_grad_()
        gc = g.clone().requires_grad_()
        aten.grid_sampler_2d(xc, gc, mode, padding, False).sum().backward()
        torch.testing.assert_close(xn.grad.cpu(), xc.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gn.grad.cpu(), gc.grad, atol=1e-3, rtol=1e-3)


# --- 3D grid_sampler ---
# grid_sampler_3d has no native torch-mlir lowering; it routes through the
# registered decomposition (trilinear + nearest). Bicubic is not defined for 3D
# by aten, so mode 2 is not tested.

MODES_3D = [0, 1]

# (input NCDHW, output (D_out, H_out, W_out)); grid is (N, D_out, H_out, W_out, 3)
CFGS_3D = [
    ((2, 3, 4, 5, 6), (3, 4, 5)),
    ((1, 1, 3, 3, 3), (3, 3, 3)),  # single channel/batch
    ((2, 4, 1, 5, 6), (2, 3, 4)),  # size-1 spatial dim
    ((1, 2, 4, 5, 6), (1, 1, 1)),  # single output point
]


def _nearest_safe_grid_3d(shape, in_shape, align_corners, dtype):
    """3D analogue of ``_nearest_safe_grid``: snap x/y/z to pixel centers.

    Nearest sampling picks a discrete voxel. A coordinate near a half-integer
    boundary can round to a different voxel on Neuron than on the fp32 CPU
    reference, so snap each axis to remove the tie.
    """
    g = torch.rand(*shape, dtype=torch.float32) * 4 - 2
    depth, height, width = in_shape[-3], in_shape[-2], in_shape[-1]

    def snap(coord, size):
        if align_corners:
            if size == 1:
                return torch.zeros_like(coord)
            src = ((coord + 1) / 2 * (size - 1)).round()
            return 2 * src / (size - 1) - 1
        src = ((coord + 1) / 2 * size - 0.5).round().clamp(0, size - 1)
        return (2 * src + 1) / size - 1

    g[..., 0] = snap(g[..., 0], width)
    g[..., 1] = snap(g[..., 1], height)
    g[..., 2] = snap(g[..., 2], depth)
    return g.to(dtype)


class TestForward3D:
    @pytest.mark.parametrize(
        "mode,padding,align", list(itertools.product(MODES_3D, PADDING_MODES, ALIGN))
    )
    @pytest.mark.parametrize("in_shape,grid_dhw", CFGS_3D)
    def test_3d_values(self, mode, padding, align, in_shape, grid_dhw):
        x = torch.randn(*in_shape)
        grid_shape = (in_shape[0], *grid_dhw, 3)
        if mode == 1:
            g = _nearest_safe_grid_3d(grid_shape, in_shape, align, torch.float32)
        else:
            g = _grid(grid_shape, torch.float32)
        expected = aten.grid_sampler_3d(x, g, mode, padding, align)
        with track_neuron_ops():
            got = aten.grid_sampler_3d(x.to("neuron"), g.to("neuron"), mode, padding, align)
            assert_op_runs_on_neuron("aten::grid_sampler_3d")
        assert got.is_contiguous()
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("mode", MODES_3D)
    def test_3d_dtypes(self, dtype, mode):
        in_shape = (2, 3, 4, 6, 7)
        x = torch.randn(*in_shape, dtype=dtype)
        if mode == 1:
            g = _nearest_safe_grid_3d((2, 3, 5, 5, 3), in_shape, False, dtype)
        else:
            g = _grid((2, 3, 5, 5, 3), dtype)
        expected = aten.grid_sampler_3d(x, g, mode, 0, False)
        with track_neuron_ops():
            got = aten.grid_sampler_3d(x.to("neuron"), g.to("neuron"), mode, 0, False)
            assert_op_runs_on_neuron("aten::grid_sampler_3d")
        assert got.dtype == dtype
        torch.testing.assert_close(got.cpu(), expected, **_tol(dtype))

    def test_3d_channels_last_matches_cpu(self):
        # aten returns a contiguous output even for a channels_last_3d input.
        x = torch.randn(2, 3, 4, 6, 7).to(memory_format=torch.channels_last_3d)
        g = _grid((2, 3, 5, 5, 3), torch.float32)
        expected = aten.grid_sampler_3d(x, g, 0, 0, False)
        with track_neuron_ops():
            got = aten.grid_sampler_3d(x.to("neuron"), g.to("neuron"), 0, 0, False)
            assert_op_runs_on_neuron("aten::grid_sampler_3d")
        assert got.is_contiguous()
        assert got.is_contiguous() == expected.is_contiguous()
        torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)

    def test_3d_out_writes_value_and_leaves_self_untouched(self):
        x = torch.randn(2, 3, 4, 6, 7)
        g = _grid((2, 3, 5, 5, 3), torch.float32)
        expected = aten.grid_sampler_3d(x, g, 0, 0, False)
        xn = x.to("neuron")
        gn = g.to("neuron")
        out = torch.empty(2, 3, 3, 5, 5, device="neuron")
        with track_neuron_ops():
            ret = aten.grid_sampler_3d.out(xn, gn, 0, 0, False, out=out)
            assert_op_runs_on_neuron("aten::grid_sampler_3d")
        torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(ret.cpu(), expected, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(xn.cpu(), x)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_3d_non_float_raises(self, dtype):
        x = torch.ones(2, 3, 4, 5, 5, dtype=dtype).to("neuron")
        g = torch.zeros(2, 3, 4, 4, 3).to("neuron")
        with pytest.raises((NotImplementedError, RuntimeError), match="grid_sampler_3d"):
            aten.grid_sampler_3d(x, g, 0, 0, False)


class TestBackwardOp3D:
    @pytest.mark.parametrize(
        "mode,padding,align", list(itertools.product(MODES_3D, PADDING_MODES, ALIGN))
    )
    def test_3d_backward_values(self, mode, padding, align):
        x = torch.randn(2, 3, 4, 5, 6)
        g = _grid((2, 3, 4, 5, 3), torch.float32)
        go = torch.randn(2, 3, 3, 4, 5)
        ei, eg = aten.grid_sampler_3d_backward(go, x, g, mode, padding, align, [True, True])
        with track_neuron_ops():
            gi, gg = aten.grid_sampler_3d_backward(
                go.to("neuron"), x.to("neuron"), g.to("neuron"), mode, padding, align, [True, True]
            )
            assert_op_runs_on_neuron("aten::grid_sampler_3d_backward")
        torch.testing.assert_close(gi.cpu(), ei, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gg.cpu(), eg, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("output_mask", [[True, False], [False, True]])
    def test_3d_backward_output_mask(self, output_mask):
        x = torch.randn(2, 3, 4, 5, 6)
        g = _grid((2, 3, 4, 5, 3), torch.float32)
        go = torch.randn(2, 3, 3, 4, 5)
        ei, eg = aten.grid_sampler_3d_backward(go, x, g, 0, 0, False, output_mask)
        with track_neuron_ops():
            gi, gg = aten.grid_sampler_3d_backward(
                go.to("neuron"), x.to("neuron"), g.to("neuron"), 0, 0, False, output_mask
            )
            assert_op_runs_on_neuron("aten::grid_sampler_3d_backward")
        if output_mask[0]:
            torch.testing.assert_close(gi.cpu(), ei, atol=1e-3, rtol=1e-3)
        if output_mask[1]:
            torch.testing.assert_close(gg.cpu(), eg, atol=1e-3, rtol=1e-3)


class TestAutograd3D:
    @pytest.mark.parametrize("mode", MODES_3D)
    @pytest.mark.parametrize("padding", PADDING_MODES)
    def test_3d_autograd(self, mode, padding):
        x = torch.randn(2, 3, 4, 5, 6)
        g = _grid((2, 3, 4, 5, 3), torch.float32)
        xn = x.clone().to("neuron").requires_grad_()
        gn = g.clone().to("neuron").requires_grad_()
        with track_neuron_ops():
            out = aten.grid_sampler_3d(xn, gn, mode, padding, False)
            out.sum().backward()
            # The forward records grid_sampler_3d, whose backward dispatches the
            # separate grid_sampler_3d_backward op; assert IT ran on Neuron.
            assert_op_runs_on_neuron("aten::grid_sampler_3d_backward")
        xc = x.clone().requires_grad_()
        gc = g.clone().requires_grad_()
        aten.grid_sampler_3d(xc, gc, mode, padding, False).sum().backward()
        torch.testing.assert_close(xn.grad.cpu(), xc.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gn.grad.cpu(), gc.grad, atol=1e-3, rtol=1e-3)
