"""Eager tests for aten::im2col / aten::col2im (F.unfold / F.fold) on Neuron.

CPU is the reference throughout. im2col only moves values, so its checks are
bit-exact. col2im accumulates overlapping windows, so the payloads are small
integers and the checks stay bit-exact (a dropped add reduction changes an
integer sum by a whole unit). See the compile-path suite in
tests/neuron_dynamo_backend/unit/test_decompositions.py for the same math under
torch.compile.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

aten = torch.ops.aten

pytestmark = pytest.mark.skipif(
    torch_neuronx.device_count() == 0, reason="No Neuron devices available"
)

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# (shape, kernel, dilation, padding, stride)
IM2COL_CFGS = [
    ((2, 3, 8, 8), [2, 2], [1, 1], [0, 0], [2, 2]),  # kernel == stride, fast path
    ((1, 4, 12, 9), [4, 3], [1, 1], [0, 0], [4, 3]),  # non-square, fast path
    ((1, 1, 4, 4), [2, 2], [1, 1], [0, 0], [1, 1]),  # overlapping, gather path
    ((2, 3, 8, 8), [3, 3], [1, 1], [1, 1], [2, 2]),  # stride + padding, gather
    ((1, 2, 8, 8), [3, 3], [2, 2], [0, 0], [1, 1]),  # dilation, gather
    ((3, 6, 6), [3, 3], [1, 1], [0, 0], [1, 1]),  # unbatched (C, H, W), gather
]

# (output_size, kernel, dilation, padding, stride)
COL2IM_CFGS = [
    ([4, 4], [2, 2], [1, 1], [0, 0], [2, 2]),  # non-overlapping
    ([5, 5], [3, 3], [1, 1], [0, 0], [1, 1]),  # heavy overlap, up to 9 taps
    ([6, 8], [2, 3], [1, 1], [0, 0], [1, 1]),  # non-square, overlap
    ([8, 8], [3, 3], [1, 1], [1, 1], [2, 2]),  # stride + padding
]


class TestIm2colForward:
    """im2col only moves values, so every check is bit-exact."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("shape,kernel,dilation,padding,stride", IM2COL_CFGS)
    def test_values(self, dtype, shape, kernel, dilation, padding, stride):
        x = torch.randn(*shape, dtype=dtype)
        expected = aten.im2col(x, kernel, dilation, padding, stride)
        with track_neuron_ops():
            got = aten.im2col(x.to("neuron"), kernel, dilation, padding, stride)
            assert_op_runs_on_neuron("aten::im2col")
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    def test_rejects_integral_like_aten(self, dtype):
        x = torch.arange(64, dtype=dtype).reshape(1, 1, 8, 8)
        with pytest.raises(NotImplementedError, match="not implemented"):
            aten.im2col(x.to("neuron"), [2, 2], [1, 1], [0, 0], [2, 2])


class TestCol2imForward:
    """col2im sums overlapping windows. Integer payloads keep the check exact."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("output_size,kernel,dilation,padding,stride", COL2IM_CFGS)
    def test_values(self, dtype, output_size, kernel, dilation, padding, stride):
        def out_dim(size, k, d, p, s):
            return (size + 2 * p - d * (k - 1) - 1) // s + 1

        ho = out_dim(output_size[0], kernel[0], dilation[0], padding[0], stride[0])
        wo = out_dim(output_size[1], kernel[1], dilation[1], padding[1], stride[1])
        # Bound the payload so every accumulated sum stays exact in bf16 (< 256).
        cols = torch.randint(-4, 5, (1, kernel[0] * kernel[1], ho * wo)).to(dtype)
        expected = aten.col2im(cols, output_size, kernel, dilation, padding, stride)
        with track_neuron_ops():
            got = aten.col2im(cols.to("neuron"), output_size, kernel, dilation, padding, stride)
            assert_op_runs_on_neuron("aten::col2im")
        assert got.dtype == dtype
        assert torch.equal(got.cpu(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_values_unbatched(self, dtype):
        # aten accepts an unbatched 2D (C*k*k, L) input and returns a 3D image.
        output_size, kernel = [5, 5], [3, 3]
        dilation, padding, stride = [1, 1], [0, 0], [1, 1]
        # stride 1, no padding, unit dilation: L = (5 - 3 + 1) ** 2 = 9.
        cols = torch.randint(-4, 5, (kernel[0] * kernel[1], 3 * 3)).to(dtype)
        expected = aten.col2im(cols, output_size, kernel, dilation, padding, stride)
        with track_neuron_ops():
            got = aten.col2im(cols.to("neuron"), output_size, kernel, dilation, padding, stride)
            assert_op_runs_on_neuron("aten::col2im")
        assert got.dtype == dtype
        assert got.ndim == 3
        assert torch.equal(got.cpu(), expected)
