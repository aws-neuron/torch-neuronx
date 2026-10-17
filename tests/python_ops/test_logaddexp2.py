"""Unit tests for the aten::logaddexp2 eager lowering.

logaddexp2(a, b) = log2(2**a + 2**b), computed in a stable form. CPU is the
reference oracle. Every device test asserts the op runs on Neuron.
"""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

OP = "aten::logaddexp2"

# neuron widens float64 to float32, so double is a known platform gap (tested on
# CPU-only paths / xfailed upstream). The device tests use the native float dtypes.
FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16]

SHAPES = [
    pytest.param((), id="0d"),
    pytest.param((1,), id="single"),
    pytest.param((128,), id="1d_small"),
    pytest.param((7, 19), id="2d_odd"),
    pytest.param((32, 64), id="2d"),
    pytest.param((4, 8, 16), id="3d"),
    pytest.param((2, 3, 4, 5), id="4d"),
    pytest.param((1, 64), id="size1_dim"),
    pytest.param((4096,), id="1d_medium"),
    pytest.param((1 << 20,), id="1d_large"),
]


def _rand(shape, dtype):
    # Spread over a wide exponent range so the stable max-shift path is exercised.
    return (torch.randn(shape) * 20).to(dtype)


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestLogaddexp2:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, dtype):
        a_cpu = _rand(shape, dtype)
        b_cpu = _rand(shape, dtype)
        expected = torch.logaddexp2(a_cpu, b_cpu)

        with track_neuron_ops():
            result = torch.logaddexp2(a_cpu.to("neuron"), b_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(OP)

    @pytest.mark.parametrize(
        "shape_a,shape_b",
        [
            pytest.param((3, 1), (1, 4), id="row_col"),
            pytest.param((5,), (3, 5), id="rank_promote"),
            pytest.param((2, 1, 4), (3, 4), id="mid_broadcast"),
            pytest.param((), (4, 4), id="scalar_tensor"),
        ],
    )
    def test_broadcasting(self, shape_a, shape_b):
        a_cpu = _rand(shape_a, torch.float32)
        b_cpu = _rand(shape_b, torch.float32)
        expected = torch.logaddexp2(a_cpu, b_cpu)

        with track_neuron_ops():
            result = torch.logaddexp2(a_cpu.to("neuron"), b_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(OP)

    def test_numeric_edges(self):
        inf, nan = float("inf"), float("nan")
        a_cpu = torch.tensor([inf, -inf, inf, -inf, 0.0, 1.0, nan, inf, 1e30, -1e30, 0.0, -0.0])
        b_cpu = torch.tensor([-inf, -inf, inf, 0.0, 0.0, 2.0, 1.0, nan, 1e30, 1e30, -0.0, 0.0])
        expected = torch.logaddexp2(a_cpu, b_cpu)

        with track_neuron_ops():
            result = torch.logaddexp2(a_cpu.to("neuron"), b_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
            assert_op_runs_on_neuron(OP)

    def test_empty(self):
        a_cpu = torch.empty((0,))
        b_cpu = torch.empty((0,))
        expected = torch.logaddexp2(a_cpu, b_cpu)
        with track_neuron_ops():
            result = torch.logaddexp2(a_cpu.to("neuron"), b_cpu.to("neuron"))
            torch.testing.assert_close(result.cpu(), expected)
            assert_op_runs_on_neuron(OP)

    def test_fp64_downcast(self):
        # float64 is widened to float32 on device. Compare against the float32
        # CPU oracle (the float64 CPU value is not reachable on Neuron).
        a64 = _rand((256,), torch.float64)
        b64 = _rand((256,), torch.float64)
        expected = torch.logaddexp2(a64.float(), b64.float())
        with track_neuron_ops():
            result = torch.logaddexp2(a64.to("neuron"), b64.to("neuron"))
            torch.testing.assert_close(result.float().cpu(), expected)
            assert_op_runs_on_neuron(OP)

    def test_out(self):
        a_cpu = _rand((64,), torch.float32)
        b_cpu = _rand((64,), torch.float32)
        expected = torch.logaddexp2(a_cpu, b_cpu)

        with track_neuron_ops():
            a_n = a_cpu.to("neuron")
            b_n = b_cpu.to("neuron")
            out = torch.empty(64, device="neuron")
            ret = torch.logaddexp2(a_n, b_n, out=out)
            assert ret is out
            torch.testing.assert_close(out.cpu(), expected)
            # self (the first operand) must be untouched.
            torch.testing.assert_close(a_n.cpu(), a_cpu)
            assert_op_runs_on_neuron(OP)

    def test_out_noncontiguous(self):
        a_cpu = _rand((8, 16), torch.float32)
        b_cpu = _rand((8, 16), torch.float32)
        expected = torch.logaddexp2(a_cpu, b_cpu)

        with track_neuron_ops():
            a_n = a_cpu.to("neuron")
            b_n = b_cpu.to("neuron")
            # A transposed view is non-contiguous.
            out = torch.empty((16, 8), device="neuron").t()
            assert not out.is_contiguous()
            ret = torch.logaddexp2(a_n, b_n, out=out)
            assert ret is out
            torch.testing.assert_close(out.cpu(), expected)
            assert_op_runs_on_neuron(OP)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_autograd(self, dtype):
        a_cpu = _rand((3, 32), dtype).requires_grad_(True)
        b_cpu = _rand((3, 32), dtype).requires_grad_(True)
        out_cpu = torch.logaddexp2(a_cpu, b_cpu)
        out_cpu.sum().backward()

        with track_neuron_ops():
            a_n = a_cpu.detach().clone().to("neuron").requires_grad_(True)
            b_n = b_cpu.detach().clone().to("neuron").requires_grad_(True)
            out_n = torch.logaddexp2(a_n, b_n)
            assert_op_runs_on_neuron(OP)
            out_n.sum().backward()

        torch.testing.assert_close(out_n.cpu(), out_cpu.detach())
        torch.testing.assert_close(a_n.grad.cpu(), a_cpu.grad)
        torch.testing.assert_close(b_n.grad.cpu(), b_cpu.grad)
