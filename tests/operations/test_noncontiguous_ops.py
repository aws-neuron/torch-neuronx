"""Tests for non-contiguous tensor handling in MLIR kernel execution.

Validates that ops produce correct results when execution tensors (outputs)
are non-contiguous — inplace ops, regular ops with transposed/sliced inputs.
"""

import torch

import torch_neuronx


def _make_noncontiguous(t):
    """Create a non-contiguous view via slicing."""
    big = torch.empty(t.shape[0] * 2, *t.shape[1:], device=t.device, dtype=t.dtype)
    big[::2] = t
    return big[::2]


def test_inplace_add_noncontiguous():
    x = torch.randn(4, 4, device="neuron:0")
    x_nc = _make_noncontiguous(x)
    expected = x_nc.to("cpu").add_(1.0)
    x_nc.add_(1.0)
    torch.testing.assert_close(x_nc.to("cpu"), expected)


def test_inplace_mul_noncontiguous():
    x = torch.randn(4, 4, device="neuron:0")
    x_nc = _make_noncontiguous(x)
    expected = x_nc.to("cpu").mul_(2.0)
    x_nc.mul_(2.0)
    torch.testing.assert_close(x_nc.to("cpu"), expected)


def test_inplace_relu_transposed():
    x = torch.randn(4, 8, device="neuron:0").t()
    assert not x.is_contiguous()
    expected = x.to("cpu").relu_()
    x.relu_()
    torch.testing.assert_close(x.to("cpu"), expected)


def test_div_transposed_by_scalar():
    base = torch.randn(2, 3, device="neuron:0")
    base_t = base.t()
    assert not base_t.is_contiguous()
    divisor = torch.tensor(2.0, device="neuron:0")
    result = torch.div(base_t, divisor)
    expected = torch.div(base_t.to("cpu"), divisor.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_matmul_transposed():
    a = torch.randn(4, 8, device="neuron:0").t()
    b = torch.randn(4, 3, device="neuron:0")
    assert not a.is_contiguous()
    result = torch.matmul(a, b)
    expected = torch.matmul(a.to("cpu"), b.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected, atol=1e-5, rtol=1e-5)


def test_add_sliced():
    x = torch.randn(8, 4, device="neuron:0")
    xs = x[::2]
    assert not xs.is_contiguous()
    result = torch.add(xs, xs)
    expected = torch.add(xs.to("cpu"), xs.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_neg_noncontiguous():
    x = torch.randn(4, 4, device="neuron:0")
    x_nc = _make_noncontiguous(x)
    result = torch.neg(x_nc)
    expected = torch.neg(x_nc.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_out_noncontiguous():
    """Test out= parameter with non-contiguous output tensor."""
    x = torch.randn(4, 4, device="neuron:0")
    y = torch.randn(4, 4, device="neuron:0")
    out = torch.empty(8, 4, device="neuron:0")[::2]  # non-contiguous
    assert not out.is_contiguous()
    torch.add(x, y, out=out)
    expected = torch.add(x.to("cpu"), y.to("cpu"))
    torch.testing.assert_close(out.to("cpu"), expected)


def test_out_transposed():
    """Test out= parameter with transposed output tensor."""
    x = torch.randn(3, 4, device="neuron:0")
    y = torch.randn(3, 4, device="neuron:0")
    out_base = torch.empty(4, 3, device="neuron:0")
    out = out_base.t()
    assert not out.is_contiguous()
    torch.add(x, y, out=out)
    expected = torch.add(x.to("cpu"), y.to("cpu"))
    torch.testing.assert_close(out.to("cpu"), expected)


def test_out_dtype_mismatch_noncontiguous():
    """Test out= with both dtype mismatch and non-contiguous output."""
    x = torch.randn(4, 4, device="neuron:0", dtype=torch.float32)
    y = torch.randn(4, 4, device="neuron:0", dtype=torch.float32)
    out_base = torch.empty(8, 4, device="neuron:0", dtype=torch.float64)
    out = out_base[::2]  # non-contiguous float64
    assert not out.is_contiguous()
    assert out.dtype == torch.float64
    torch.add(x, y, out=out)
    expected = torch.add(x.to("cpu"), y.to("cpu")).to(torch.float64)
    torch.testing.assert_close(out.to("cpu"), expected)


def _make_broadcast(shape, broadcast_dims, device="neuron:0", dtype=torch.float32):
    """Create a broadcast tensor with stride=0 in specified dimensions.

    Args:
        shape: desired broadcast shape, e.g. (3, 4, 5)
        broadcast_dims: dims to broadcast, e.g. [0] means dim 0 has stride=0
    """
    base_shape = list(shape)
    for d in broadcast_dims:
        base_shape[d] = 1
    if dtype.is_floating_point:
        base = torch.randn(base_shape, device=device, dtype=dtype)
    else:
        base = torch.randint(0, 100, base_shape, device=device, dtype=dtype)
    return base.expand(shape)


def test_broadcast_add_stride0():
    """Test binary op with broadcast tensor (stride=0 in dim 0)."""
    a = _make_broadcast((4, 4), [0])
    b = torch.randn(4, 4, device="neuron:0")
    assert not a.is_contiguous()
    result = torch.add(a, b)
    expected = torch.add(a.to("cpu"), b.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_broadcast_multi_dim_stride0():
    """Test with multiple broadcast dimensions (3D, dims 0 and 2)."""
    a = _make_broadcast((3, 4, 5), [0, 2])
    b = torch.randn(3, 4, 5, device="neuron:0")
    assert not a.is_contiguous()
    result = torch.add(a, b)
    expected = torch.add(a.to("cpu"), b.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_broadcast_bitwise_and():
    """Test integer op with broadcast tensor."""
    a = _make_broadcast((5, 5), [0], dtype=torch.int32)
    b = torch.randint(0, 100, (5, 5), device="neuron:0", dtype=torch.int32)
    assert not a.is_contiguous()
    result = torch.bitwise_and(a, b)
    expected = torch.bitwise_and(a.to("cpu"), b.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_broadcast_unary_signbit():
    """Test unary op with broadcast tensor."""
    a = _make_broadcast((5, 6, 2), [0], dtype=torch.float32)
    assert not a.is_contiguous()
    result = torch.signbit(a)
    expected = torch.signbit(a.to("cpu"))
    torch.testing.assert_close(result.to("cpu"), expected)


def test_cat_with_broadcast_input():
    """Test torch.cat with a broadcast (stride=0) tensor."""
    a = torch.randn(1, 32, 1, 42, device="neuron:0")
    b = a.expand(8, 32, 32, 42)
    c = torch.randn(8, 32, 32, 44, device="neuron:0")
    assert not b.is_contiguous()
    result = torch.cat([c, b], dim=-1)
    expected = torch.cat([c.to("cpu"), b.to("cpu")], dim=-1)
    torch.testing.assert_close(result.to("cpu"), expected)


def test_broadcast_from_slice():
    """Test broadcast tensor derived from slice (non-contiguous storage).

    This pattern was failing: slice → unsqueeze → broadcast → op
    The slice creates stride [2], then broadcast creates strides [2, 0].
    The non-contiguous base stride caused MLIR kernel rejection.
    """
    original = torch.randn(4, device="neuron:0")
    sliced = original[::2]  # stride [2], not contiguous
    broadcast = sliced.unsqueeze(1).expand(2, 5)  # strides [2, 0]
    assert not broadcast.is_contiguous()
    result = broadcast + 1.0
    expected = broadcast.to("cpu") + 1.0
    torch.testing.assert_close(result.to("cpu"), expected)
