"""E2E tests for foreach ops (functional + out= overloads)."""

import pytest
import torch

SHAPES = [
    pytest.param((1,), id="scalar-1d"),
    pytest.param((64,), id="1d"),
    pytest.param((8, 17), id="2d-odd"),
    pytest.param((2, 3, 4), id="3d"),
    pytest.param((128, 128), id="large-2d"),
]

# Reduced shape set for the `out=` overloads: every param is a separate compile.
OUT_SHAPES = [pytest.param((64,), id="1d"), pytest.param((8, 17), id="2d-odd")]

FLOAT_DTYPES = [
    pytest.param(torch.float32, id="f32"),
    pytest.param(torch.float16, id="f16"),
    pytest.param(torch.bfloat16, id="bf16"),
]

ALL_DTYPES = [
    *FLOAT_DTYPES,
    pytest.param(torch.int32, id="i32"),
    pytest.param(torch.int64, id="i64"),
]

# Integer scalars are valid for both float and integer tensors (no promotion).
SCALARS = [-1, 0, 2]

# (dst_dtype, src_dtype): same-dtype for each dtype, plus mixed-precision copies
# in both directions (widening and narrowing, since copy_ casts src to dst).
COPY_DTYPES = [
    pytest.param(torch.float32, torch.float32, id="f32-f32"),
    pytest.param(torch.float16, torch.float16, id="f16-f16"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="bf16-bf16"),
    pytest.param(torch.int32, torch.int32, id="i32-i32"),
    pytest.param(torch.int64, torch.int64, id="i64-i64"),
    pytest.param(torch.float32, torch.bfloat16, id="f32<-bf16"),
    pytest.param(torch.bfloat16, torch.float32, id="bf16<-f32"),
    pytest.param(torch.float16, torch.float32, id="f16<-f32"),
    pytest.param(torch.int32, torch.int64, id="i32<-i64"),
    pytest.param(torch.int64, torch.int32, id="i64<-i32"),
]


def _make_list(shape, dtype, n, seed=0):
    torch.manual_seed(seed)
    if dtype in (torch.int32, torch.int64):
        return [torch.randint(-128, 128, shape, dtype=dtype) for _ in range(n)]
    return [torch.randn(shape, dtype=dtype) for _ in range(n)]


def _to_neuron(ts):
    return [t.to("neuron") for t in ts]


def _assert_list_close(neuron_out, cpu_out, dtype):
    if dtype in (torch.float16, torch.bfloat16):
        rtol, atol = 1e-2, 1e-3
    elif dtype in (torch.int32, torch.int64):
        rtol, atol = 0, 0
    else:
        rtol, atol = 1e-4, 1e-5
    for n, c in zip(neuron_out, cpu_out, strict=True):
        torch.testing.assert_close(n.cpu(), c, rtol=rtol, atol=atol)


class TestForeachZeroE2E:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_foreach_zero_(self, shape, dtype):
        xs = _make_list(shape, dtype, 3, seed=0)

        def fn(a, b, c):
            torch._foreach_zero_([a, b, c])
            return a, b, c

        with torch.no_grad():
            cpu_out = fn(*[t.clone() for t in xs])
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(*_to_neuron(xs))

        _assert_list_close(list(neuron_out), list(cpu_out), dtype)


class TestForeachCopyE2E:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dst_dtype,src_dtype", COPY_DTYPES)
    def test_foreach_copy_(self, shape, dst_dtype, src_dtype):
        xs = _make_list(shape, dst_dtype, 3, seed=0)
        src = _make_list(shape, src_dtype, 3, seed=1)

        def fn(a, b, c, sa, sb, sc):
            torch._foreach_copy_([a, b, c], [sa, sb, sc])
            return a, b, c

        with torch.no_grad():
            cpu_out = fn(*[t.clone() for t in xs], *[t.clone() for t in src])
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(*_to_neuron(xs), *_to_neuron(src))

        _assert_list_close(list(neuron_out), list(cpu_out), dst_dtype)


class TestForeachMaxMinScalarE2E:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_foreach_maximum_scalar(self, shape, dtype, scalar):
        xs = _make_list(shape, dtype, 3, seed=0)

        def fn(a, b, c):
            return torch._foreach_maximum([a, b, c], scalar)

        with torch.no_grad():
            cpu_out = fn(*xs)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(*_to_neuron(xs))

        _assert_list_close(neuron_out, cpu_out, dtype)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("scalar", SCALARS)
    def test_foreach_minimum_scalar(self, shape, dtype, scalar):
        xs = _make_list(shape, dtype, 3, seed=0)

        def fn(a, b, c):
            return torch._foreach_minimum([a, b, c], scalar)

        with torch.no_grad():
            cpu_out = fn(*xs)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(*_to_neuron(xs))

        _assert_list_close(neuron_out, cpu_out, dtype)

    @pytest.mark.parametrize("shape", OUT_SHAPES)
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_foreach_maximum_scalar_out(self, shape, dtype):
        xs = _make_list(shape, dtype, 3, seed=0)

        def fn(a, b, c):
            outs = [torch.zeros_like(t) for t in (a, b, c)]
            torch.ops.aten._foreach_maximum.Scalar_out([a, b, c], 0, out=outs)
            return outs[0], outs[1], outs[2]

        with torch.no_grad():
            cpu_out = fn(*xs)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(*_to_neuron(xs))

        _assert_list_close(list(neuron_out), list(cpu_out), dtype)

    @pytest.mark.parametrize("shape", OUT_SHAPES)
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_foreach_minimum_scalar_out(self, shape, dtype):
        xs = _make_list(shape, dtype, 3, seed=0)

        def fn(a, b, c):
            outs = [torch.zeros_like(t) for t in (a, b, c)]
            torch.ops.aten._foreach_minimum.Scalar_out([a, b, c], 0, out=outs)
            return outs[0], outs[1], outs[2]

        with torch.no_grad():
            cpu_out = fn(*xs)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(*_to_neuron(xs))

        _assert_list_close(list(neuron_out), list(cpu_out), dtype)
