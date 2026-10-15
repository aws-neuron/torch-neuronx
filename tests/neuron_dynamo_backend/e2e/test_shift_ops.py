"""
E2E tests for bitwise shift ops (__rshift__ and __lshift__, Tensor and Scalar variants).
"""

import pytest
import torch

SHAPES = [
    pytest.param((64,), id="1d"),
    pytest.param((16, 32), id="2d"),
    pytest.param((8, 17), id="2d-odd"),
    pytest.param((8, 16, 32), id="3d"),
    pytest.param((256, 256), id="large-2d"),
]

DTYPES = [
    pytest.param(torch.int32, id="i32"),
    pytest.param(torch.int64, id="i64"),
]

SHIFT_AMOUNTS = [1, 4, 8]

# Reduced matrix for the `out=` overloads on purpose: shape/dtype coverage is
# the functional tests' job, and every param here is a separate Neuron compile.
OUT_SHAPES = [pytest.param((64,), id="1d"), pytest.param((8, 17), id="2d-odd")]


class TestRshiftTensorE2E:
    """E2E tests for aten.__rshift__.Tensor (tensor >> tensor)."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", SHIFT_AMOUNTS)
    def test_rshift_tensor(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)
        # Shift amount must be non-negative and < bitwidth
        s = torch.full(shape, shift, dtype=dtype)

        def fn(a, b):
            return a >> b

        with torch.no_grad():
            cpu_out = fn(x, s)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), s.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)


class TestLshiftE2E:
    """E2E tests for aten.__lshift__ (tensor << tensor, tensor << int).

    Only aten::__rshift__.Tensor has a StableHLO conversion pattern, so the
    left shifts reach the compiler via a decomposition to
    bitwise_left_shift. Without it these fail to lower.
    """

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", SHIFT_AMOUNTS)
    def test_lshift_tensor(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)
        s = torch.full(shape, shift, dtype=dtype)

        def fn(a, b):
            return a << b

        with torch.no_grad():
            cpu_out = fn(x, s)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), s.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", SHIFT_AMOUNTS)
    def test_lshift_scalar(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)

        def fn(a):
            return a << shift

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)


class TestShiftOutOverloadE2E:
    """The `out=` overloads need no decomposition of their own.

    `aten.__lshift__.Tensor_out` / `.Scalar_out` are never registered in
    `decompositions.py`, and they do not need to be: AOTAutograd's
    functionalization rewrites an `out=` call to its functional form before the
    decomposition table is consulted, so the graph the backend sees contains
    `aten.__lshift__.Tensor` / `.Scalar` and hits the two functional
    registrations. These tests pin that behaviour -- if functionalization ever
    stopped stripping the out variant, they would fail with
    `Failed Torch-MLIR Lowering` and the out overloads would then need
    registering.
    """

    @pytest.mark.parametrize("shape", OUT_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", [4])
    def test_lshift_tensor_out(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)
        s = torch.full(shape, shift, dtype=dtype)

        def fn(a, b):
            z = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
            torch.ops.aten.__lshift__.Tensor_out(a, b, out=z)
            return z

        with torch.no_grad():
            cpu_out = fn(x, s)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), s.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize("shape", OUT_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", [4])
    def test_lshift_scalar_out(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)

        def fn(a):
            z = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
            torch.ops.aten.__lshift__.Scalar_out(a, shift, out=z)
            return z

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)

    @pytest.mark.parametrize("shape", OUT_SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", [4])
    def test_rshift_tensor_out(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)
        s = torch.full(shape, shift, dtype=dtype)

        def fn(a, b):
            z = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
            torch.ops.aten.__rshift__.Tensor_out(a, b, out=z)
            return z

        with torch.no_grad():
            cpu_out = fn(x, s)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"), s.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)


class TestRshiftScalarE2E:
    """E2E tests for aten.__rshift__.Scalar (tensor >> int)."""

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shift", SHIFT_AMOUNTS)
    def test_rshift_scalar(self, shape, dtype, shift):
        torch.manual_seed(42)
        x = torch.randint(0, 1024, shape, dtype=dtype)

        def fn(a):
            return a >> shift

        with torch.no_grad():
            cpu_out = fn(x)
            compiled_fn = torch.compile(fn, backend="neuron", fullgraph=True)
            neuron_out = compiled_fn(x.to("neuron"))

        torch.testing.assert_close(neuron_out.cpu(), cpu_out)
