"""Tests for aten::minimum and aten::maximum on Neuron."""

import pytest
import torch

import torch_neuronx
from tests.utils.neuron_test_utils import assert_op_runs_on_neuron, track_neuron_ops

DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int32]
SHAPES = [(8,), (4, 8), (2, 3, 4)]


@pytest.mark.skipif(torch_neuronx.device_count() == 0, reason="No Neuron devices available")
class TestMinimumMaximum:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "op,aten_name",
        [
            (torch.minimum, "aten::minimum"),
            (torch.maximum, "aten::maximum"),
        ],
    )
    def test_basic(self, op, aten_name, shape, dtype):
        """Output matches CPU reference."""
        a = torch.randn(*shape).to(dtype)
        b = torch.randn(*shape).to(dtype)

        with track_neuron_ops():
            out = op(a.to("neuron"), b.to("neuron"))
            assert_op_runs_on_neuron(aten_name)

        torch.testing.assert_close(out.cpu(), op(a, b), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "op,aten_name",
        [
            (torch.minimum, "aten::minimum"),
            (torch.maximum, "aten::maximum"),
        ],
    )
    def test_broadcasting(self, op, aten_name):
        """Supports broadcasting."""
        a = torch.randn(4, 1)
        b = torch.randn(1, 8)

        with track_neuron_ops():
            out = op(a.to("neuron"), b.to("neuron"))
            assert_op_runs_on_neuron(aten_name)

        torch.testing.assert_close(out.cpu(), op(a, b), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "op,aten_name",
        [
            (torch.minimum, "aten::minimum"),
            (torch.maximum, "aten::maximum"),
        ],
    )
    def test_out(self, op, aten_name):
        """Supports preallocated out tensor."""
        a = torch.randn(8)
        b = torch.randn(8)
        out = torch.empty(8, device="neuron")

        with track_neuron_ops():
            op(a.to("neuron"), b.to("neuron"), out=out)
            assert_op_runs_on_neuron(aten_name)

        torch.testing.assert_close(out.cpu(), op(a, b), rtol=0, atol=0)
